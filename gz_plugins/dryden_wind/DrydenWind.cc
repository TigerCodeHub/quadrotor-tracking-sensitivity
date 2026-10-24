// DrydenWind: a Gazebo Harmonic (gz-sim8) world system plugin that applies a
// Dryden-FORM atmospheric turbulence gust (MIL-F-8785C / MIL-STD-1797A spectral form; the
// gust SPECTRUM is validated, but the force coupling below is the stock approximation, not a
// physical aerodynamic model) to every link flagged
// <enable_wind>true</enable_wind>, using the SAME force model as the stock
// WindEffects system:
//
//     F_world = link_mass * force_scaling_factor * (wind_vel - link_vel)
//
// so a white-noise arm (stock WindEffects) and a Dryden arm (this plugin) differ
// only in the wind-velocity time series, never in the wind->force coupling (both apply the
// mass-proportional, velocity-linear stock law above). NOTE the two time series differ in
// several ways at once -- spectrum, deterministic content, and anisotropy -- so a paired
// white-vs-Dryden contrast bounds the turbulence-model effect rather than isolating the spectrum.
//
// Turbulence model: Dryden forming filters (MIL-F-8785C / MIL-STD-1797A), driven
// by unit-PSD Gaussian white noise, in wind axes (x = airspeed direction):
//   u (longitudinal): H_u(s) = sigma_u * sqrt(2 Lu/(pi V)) / (1 + (Lu/V) s)
//   v (lateral):      H_v(s) = sigma_v * sqrt(  Lv/(pi V)) * (1 + sqrt(3)(Lv/V)s)
//                                                          / (1 + (Lv/V) s)^2
//   w (vertical):     H_w(s)  (same 2nd-order form as v, with Lw, sigma_w)
// Output variance of each filter under unit white noise = sigma^2 (Dryden norm).
//
// Discretisation:
//   u : EXACT Ornstein-Uhlenbeck update (phi = exp(-dt/tau)), stationary std = sigma_u.
//   v,w : Euler-Maruyama of the controllable-canonical 2-state realisation of the
//         forming filter. Accurate while dt << tau = L/V (always true here: at
//         V<=25 m/s, L>=~5 m => tau>=0.2 s, sim dt ~1-4 ms => dt/tau <~0.02); the
//         plugin sub-steps if dt/tau exceeds 0.05 to keep the spectrum exact.
//
// Length scales & intensities (low altitude, h in ft, 0 < h < 1000; MIL-F-8785C):
//   Lw = h ;  Lu = Lv = h / (0.177 + 0.000823 h)^1.2
//   sigma_w = 0.1 * W20 ;  sigma_u = sigma_v = sigma_w / (0.177 + 0.000823 h)^0.4
// where W20 is the wind speed at 20 ft. Either supply <w20> (true MIL-STD severity,
// altitude-dependent sigma) OR supply explicit <sigma>su sv sw</sigma> (used to match
// a white-noise arm's per-axis RMS). Length scales always come from the altitude
// formula unless <length_scale> is given (fixed mode).
//
// SDF params (all optional unless noted):
//   <mean_wind>x y z</mean_wind>        world-frame mean wind, m/s        (default 0 0 0)
//   <w20>V</w20>                        wind at 20 ft, m/s -> MIL-STD sigma(h)
//   <sigma>su sv sw</sigma>             explicit per-axis intensities, m/s (overrides w20)
//   <length_scale>Lu Lv Lw</length_scale>  fixed scales, m (else altitude formula)
//   <force_scaling_factor>k</force_scaling_factor>  default 1
//   <min_airspeed>V</min_airspeed>      floor for V in tau=L/V, m/s       (default 2)
//   <axis_frame>airspeed|world_meanwind</axis_frame>  gust axes (default airspeed).
//       world_meanwind = fixed frame: u along <axis_heading_deg>, w along world up
//       (use to match a white-noise arm's axes for a clean spectrum-only contrast).
//   <axis_heading_deg>deg</axis_heading_deg>  u-axis heading for world_meanwind (default 0)
//   <seed>N</seed>                      RNG seed                          (default 0 -> random)
//   <log_file>path</log_file>           append t,entity,u,v,w,V,h_ft,Lu,Lv,Lw,windx,windy,windz
//
// Build: see CMakeLists.txt. Load as a <world> plugin (like WindEffects).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <unordered_map>

#include <gz/plugin/Register.hh>
#include <gz/math/Vector3.hh>
#include <gz/math/Pose3.hh>

#include <gz/sim/System.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Link.hh>
#include <gz/sim/components/WindMode.hh>
#include <gz/sim/components/Inertial.hh>
#include <gz/sim/components/Name.hh>

namespace tdr
{
using namespace gz;
using namespace gz::sim;

// Per-link turbulence filter state.
struct GustState
{
  double u{0.0};             // OU longitudinal gust (wind-axis x)
  double v1{0.0}, v2{0.0};   // 2-state lateral filter
  double w1{0.0}, w2{0.0};   // 2-state vertical filter
  std::mt19937_64 rng;       // independent stream per link
  bool inited{false};
};

class DrydenWind
  : public System,
    public ISystemConfigure,
    public ISystemPreUpdate
{
  public: void Configure(const Entity &,
                         const std::shared_ptr<const sdf::Element> &_sdf,
                         EntityComponentManager &,
                         EventManager &) override
  {
    auto getVec = [&](const std::string &tag, math::Vector3d def)
    {
      if (_sdf->HasElement(tag))
        return _sdf->Get<math::Vector3d>(tag);
      return def;
    };
    this->meanWind = getVec("mean_wind", math::Vector3d::Zero);

    if (_sdf->HasElement("sigma"))
    {
      this->sigma = _sdf->Get<math::Vector3d>("sigma");
      this->useExplicitSigma = true;
    }
    if (_sdf->HasElement("w20"))
      this->w20 = _sdf->Get<double>("w20");

    if (_sdf->HasElement("length_scale"))
    {
      this->fixedL = _sdf->Get<math::Vector3d>("length_scale");
      this->useFixedL = true;
    }
    if (_sdf->HasElement("force_scaling_factor"))
      this->forceScaling = _sdf->Get<double>("force_scaling_factor");
    if (_sdf->HasElement("min_airspeed"))
      this->minAirspeed = _sdf->Get<double>("min_airspeed");
    if (_sdf->HasElement("axis_frame"))
      this->worldFrame = (_sdf->Get<std::string>("axis_frame") == "world_meanwind");
    if (_sdf->HasElement("axis_heading_deg"))
      this->axisHeading = _sdf->Get<double>("axis_heading_deg") * M_PI / 180.0;
    if (_sdf->HasElement("seed"))
      this->seed = _sdf->Get<uint64_t>("seed");
    if (_sdf->HasElement("log_file"))
    {
      this->logPath = _sdf->Get<std::string>("log_file");
      this->logStream.open(this->logPath, std::ios::out | std::ios::trunc);
      if (this->logStream.is_open())
        this->logStream << "t,entity,u,v,w,V,h_ft,Lu,Lv,Lw,windx,windy,windz\n";
    }
  }

  public: void PreUpdate(const UpdateInfo &_info,
                         EntityComponentManager &_ecm) override
  {
    if (_info.paused)
      return;
    const double dt = std::chrono::duration<double>(_info.dt).count();
    if (dt <= 0.0)
      return;
    const double t = std::chrono::duration<double>(_info.simTime).count();

    _ecm.Each<components::Link, components::WindMode>(
      [&](const Entity &_entity,
          const components::Link *,
          const components::WindMode *_windMode) -> bool
      {
        if (!_windMode->Data())
          return true;   // link not wind-enabled

        Link link(_entity);
        // Make sure velocity + pose + inertia are queryable.
        if (!this->initedLinks.count(_entity))
        {
          link.EnableVelocityChecks(_ecm, true);
          GustState st;
          uint64_t s = this->seed
            ? (this->seed ^ (static_cast<uint64_t>(_entity) * 0x9E3779B97F4A7C15ull))
            : std::random_device{}();
          st.rng.seed(s);
          st.inited = true;
          this->states.emplace(_entity, std::move(st));
          this->initedLinks.insert(_entity);
          return true;   // skip first step (no velocity yet)
        }

        auto inertial = _ecm.Component<components::Inertial>(_entity);
        auto worldVelOpt = link.WorldLinearVelocity(_ecm);
        auto worldPoseOpt = link.WorldPose(_ecm);
        if (!inertial || !worldVelOpt || !worldPoseOpt)
          return true;

        const double mass = inertial->Data().MassMatrix().Mass();
        const math::Vector3d linkVel = *worldVelOpt;
        const double hMeters = std::max(0.0, worldPoseOpt->Pos().Z());
        const double hFt = std::max(3.0, hMeters * 3.28084);  // avoid Lw->0

        // Airspeed drives the frozen-field temporal scaling.
        const math::Vector3d relVel = linkVel - this->meanWind;
        double V = relVel.Length();
        if (V < this->minAirspeed) V = this->minAirspeed;

        // Length scales.
        double Lu, Lv, Lw;
        if (this->useFixedL)
        { Lu = this->fixedL.X(); Lv = this->fixedL.Y(); Lw = this->fixedL.Z(); }
        else
        {
          const double denom = std::pow(0.177 + 0.000823 * hFt, 1.2);
          const double Lw_ft = hFt;
          const double Lu_ft = hFt / denom;
          // MIL-STD scales are defined in ft; convert to metres.
          Lw = Lw_ft * 0.3048;
          Lu = Lu_ft * 0.3048;
          Lv = Lu;
        }

        // Intensities.
        double su, sv, sw;
        if (this->useExplicitSigma)
        { su = this->sigma.X(); sv = this->sigma.Y(); sw = this->sigma.Z(); }
        else
        {
          const double k = std::pow(0.177 + 0.000823 * hFt, 0.4);
          sw = 0.1 * this->w20;
          su = sw / k;
          sv = su;
        }

        GustState &st = this->states[_entity];

        // --- u: exact Ornstein-Uhlenbeck (first-order Dryden) ---
        {
          const double tau = Lu / V;
          const double phi = std::exp(-dt / tau);
          const double q = su * std::sqrt(std::max(0.0, 1.0 - phi * phi));
          st.u = phi * st.u + q * this->gauss(st.rng);
        }
        // --- v, w: Euler-Maruyama of 2-state Dryden forming filter ---
        const double gv = this->step2(st.v1, st.v2, Lv, sv, V, dt, st.rng);
        const double gw = this->step2(st.w1, st.w2, Lw, sw, V, dt, st.rng);
        const double gu = st.u;

        // Wind-axis -> world frame.
        math::Vector3d xh, yh, zh;
        if (this->worldFrame)
        {
          // Fixed world frame: u along a horizontal heading, w along world up.
          // Used to isolate the noise SPECTRUM against a white-noise arm that
          // injects its magnitude noise along the mean-wind heading and its
          // vertical noise along world z (matched axes => clean white-vs-Dryden).
          xh = math::Vector3d(std::cos(this->axisHeading),
                              std::sin(this->axisHeading), 0.0);
          zh = math::Vector3d(0, 0, 1);
          yh = zh.Cross(xh);
        }
        else
        {
          // Physical wind axes: x_hat = airspeed direction; z_hat ~ world up
          // projected perpendicular to x_hat; y_hat completes the frame. Only use
          // the airspeed direction when well-defined (>= min_airspeed).
          xh = relVel;
          if (xh.Length() < this->minAirspeed) xh = math::Vector3d(1, 0, 0);
          xh.Normalize();
          math::Vector3d up(0, 0, 1);
          zh = up - xh * up.Dot(xh);
          if (zh.Length() < 1e-6) zh = math::Vector3d(0, 0, 1);
          zh.Normalize();
          yh = zh.Cross(xh);
        }

        const math::Vector3d gustWorld = xh * gu + yh * gv + zh * gw;
        const math::Vector3d windVel = this->meanWind + gustWorld;

        const math::Vector3d force =
          mass * this->forceScaling * (windVel - linkVel);
        link.AddWorldForce(_ecm, force);

        if (this->logStream.is_open())
        {
          std::lock_guard<std::mutex> lk(this->logMutex);
          this->logStream << t << ',' << _entity << ','
            << gu << ',' << gv << ',' << gw << ',' << V << ',' << hFt << ','
            << Lu << ',' << Lv << ',' << Lw << ','
            << windVel.X() << ',' << windVel.Y() << ',' << windVel.Z() << '\n';
        }
        return true;
      });
  }

  // One Euler-Maruyama step of the 2-state Dryden lateral/vertical filter.
  // Continuous controllable-canonical form for H(s) = g (1 + sqrt3 tau s)/(1+tau s)^2
  // driven by unit-PSD white noise. The gain g is set so the OUTPUT variance is
  // exactly sigma^2 under unit-PSD input: the 2-state realisation has steady-state
  // Var(y) = g^2 * tau^3 (Lyapunov), so g = sigma * sqrt(L/V) / tau^2 gives Var=sigma^2.
  // (NB: the textbook MIL-spec sqrt(L/(pi*V)) gain assumes a noise PSD of pi, not 1;
  // driving it with unit-PSD noise as we do here would leave Var = sigma^2/pi.)
  // Returns the filter output (gust component). Sub-steps if dt/tau is large.
  private: double step2(double &x1, double &x2, double L, double sigma,
                        double V, double dt, std::mt19937_64 &rng)
  {
    const double tau = L / V;
    const double a = 1.0 / (tau * tau);   // -a on x1
    const double b = 2.0 / tau;           // -b on x2
    const double g = sigma * std::sqrt(L / V) / (tau * tau);
    const double sqrt3 = std::sqrt(3.0);

    int nsub = 1;
    if (dt / tau > 0.05)
      nsub = static_cast<int>(std::ceil((dt / tau) / 0.05));
    const double h = dt / nsub;
    const double sh = std::sqrt(h);
    for (int i = 0; i < nsub; ++i)
    {
      const double eta = this->gauss(rng);   // ~N(0,1); * sqrt(h) => dW
      const double dx1 = x2;
      const double dx2 = -a * x1 - b * x2;
      x1 += h * dx1;
      x2 += h * dx2 + sh * eta;   // unit-PSD white noise input on x2
    }
    // Output y = g*x1 + g*sqrt3*tau*x2 (numerator 1 + sqrt3 tau s).
    return g * x1 + g * sqrt3 * tau * x2;
  }

  private: double gauss(std::mt19937_64 &rng)
  {
    return this->normal(rng);
  }

  // members
  private: math::Vector3d meanWind{math::Vector3d::Zero};
  private: math::Vector3d sigma{math::Vector3d::Zero};
  private: math::Vector3d fixedL{math::Vector3d::Zero};
  private: bool useExplicitSigma{false};
  private: bool useFixedL{false};
  private: double w20{0.0};
  private: double forceScaling{1.0};
  private: double minAirspeed{2.0};
  private: uint64_t seed{0};
  private: bool worldFrame{false};   // false=airspeed axes, true=fixed world_meanwind
  private: double axisHeading{0.0};   // radians, u-axis heading in world_meanwind mode

  private: std::unordered_map<Entity, GustState> states;
  private: std::set<Entity> initedLinks;
  private: std::normal_distribution<double> normal{0.0, 1.0};

  private: std::string logPath;
  private: std::ofstream logStream;
  private: std::mutex logMutex;
};
}  // namespace tdr

GZ_ADD_PLUGIN(tdr::DrydenWind,
              gz::sim::System,
              tdr::DrydenWind::ISystemConfigure,
              tdr::DrydenWind::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(tdr::DrydenWind, "tdr::DrydenWind")
