import csv
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from tdresearch.campaign import campaign_summary, generate_baseline_campaign, generate_sobol_campaign, read_airframes
from tdresearch.generate import _normalize_mrs_motor_plugins, generate_models


AIRFRAMES = [
    {
        "airframe_id": "small",
        "display_name": "Small",
        "nominal_mass_kg": "1.0",
        "wheelbase_m": "0.4",
        "rotor_diameter_m": "0.2",
        "motor_constant_n_per_rad_s2": "0.00001",
        "upstream_max_rot_velocity_rad_s": "1000",
        "inertia_xx_kg_m2": "0.01",
        "inertia_yy_kg_m2": "0.01",
        "inertia_zz_kg_m2": "0.02",
        "parameter_confidence": "estimated",
        "source_url": "https://example.com/small",
    },
    {
        "airframe_id": "large",
        "display_name": "Large",
        "nominal_mass_kg": "3.0",
        "wheelbase_m": "0.7",
        "rotor_diameter_m": "0.4",
        "motor_constant_n_per_rad_s2": "0.00002",
        "upstream_max_rot_velocity_rad_s": "800",
        "inertia_xx_kg_m2": "0.1",
        "inertia_yy_kg_m2": "0.1",
        "inertia_zz_kg_m2": "0.2",
        "parameter_confidence": "estimated",
        "source_url": "https://example.com/large",
    },
]


class CampaignTests(unittest.TestCase):
    def test_sobol_design_is_reproducible_and_bounded(self):
        first = generate_sobol_campaign(AIRFRAMES, samples_per_airframe=16, repetitions=3, seed=9)
        second = generate_sobol_campaign(AIRFRAMES, samples_per_airframe=16, repetitions=3, seed=9)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 96)
        self.assertEqual(len({row["run_id"] for row in first}), 96)
        self.assertTrue(all(0.0 <= row["payload_mass_pct"] <= 30.0 for row in first))
        self.assertTrue(all(0.0 <= row["turbulence_rms_m_s"] <= 3.0 for row in first))
        self.assertTrue(all(0.0 <= row["mean_wind_m_s"] <= 6.0 for row in first))
        self.assertTrue(all(row["study_region"] == "primary" for row in first))
        self.assertTrue(all(1.5 <= row["thrust_to_weight"] <= 3.0 for row in first))
        self.assertTrue(all(0 <= row["wind_seed"] < 2**31 for row in first))
        self.assertTrue(all(row["com_x_m"] == 0.0 for row in first))
        self.assertTrue(all(row["com_z_m"] == 0.0 for row in first))
        self.assertTrue(any(row["com_y_m"] != 0.0 for row in first))

    def test_boundary_region_retains_original_wind_envelope(self):
        rows = generate_sobol_campaign(
            AIRFRAMES[:1],
            samples_per_airframe=64,
            repetitions=1,
            seed=9,
            region="boundary",
        )
        self.assertTrue(all(0.0 <= row["turbulence_rms_m_s"] <= 6.0 for row in rows))
        self.assertTrue(all(0.0 <= row["mean_wind_m_s"] <= 10.0 for row in rows))
        self.assertTrue(any(row["turbulence_rms_m_s"] > 3.0 for row in rows))
        self.assertTrue(any(row["mean_wind_m_s"] > 6.0 for row in rows))
        self.assertTrue(all(row["study_region"] == "boundary" for row in rows))

    def test_repeats_share_configuration_but_not_wind_seed(self):
        rows = generate_sobol_campaign(AIRFRAMES[:1], samples_per_airframe=8, repetitions=4, seed=4)
        sample = [row for row in rows if row["sample_index"] == 0]
        self.assertEqual(len({row["model_name"] for row in sample}), 1)
        self.assertEqual(len({row["wind_seed"] for row in sample}), 4)
        self.assertEqual({row["repeat"] for row in sample}, {1, 2, 3, 4})

    def test_summary_counts_configurations(self):
        rows = generate_sobol_campaign(AIRFRAMES, samples_per_airframe=8, repetitions=2)
        summary = campaign_summary(rows)
        self.assertEqual(summary["airframe_count"], 2)
        self.assertEqual(summary["configuration_count"], 16)
        self.assertEqual(summary["run_count"], 32)

    def test_baselines_preserve_upstream_limits(self):
        rows = generate_baseline_campaign(AIRFRAMES, repetitions=2)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["max_rot_velocity_rad_s"], 1000.0)
        self.assertTrue(all(row["wind_level"] == "calm" for row in rows))

    def test_catalog_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "airframes.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=AIRFRAMES[0].keys())
                writer.writeheader()
                writer.writerows(AIRFRAMES)
            self.assertEqual(len(read_airframes(path)), 2)

    def test_absolute_campaign_values_are_written_to_model(self):
        base_sdf = """<sdf version="1.9"><model name="base"><link name="base_link">
          <inertial><mass>2</mass><inertia><ixx>.02</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>.02</iyy><iyz>0</iyz><izz>.04</izz></inertia></inertial></link>
          <plugin name="motor" filename="motor"><maxRotVelocity>1000</maxRotVelocity></plugin>
        </model></sdf>"""
        row = generate_sobol_campaign(AIRFRAMES[:1], samples_per_airframe=1, repetitions=1)[0]
        row["airframe_id"] = "px4_x500"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "base.sdf"
            source.write_text(base_sdf, encoding="utf-8")
            design = root / "design.csv"
            with design.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)
            output = root / "models"
            generate_models(design, output, source, motor_sdf=source)
            model = ET.parse(output / row["model_name"] / "model.sdf").getroot()
            self.assertAlmostEqual(float(model.findtext("model/link/inertial/mass")), row["mass_kg"])
            self.assertAlmostEqual(
                float(model.findtext("model/link/inertial/inertia/ixx")), row["inertia_xx_kg_m2"]
            )
            self.assertEqual(
                float(model.findtext("model/plugin/maxRotVelocity")), row["max_rot_velocity_rad_s"]
            )

    def test_mrs_motor_plugins_are_converted_for_gazebo_harmonic(self):
        plugins = "".join(
            f"""<plugin filename="MrsGazeboCommonResources_MulticopterMotorModel"
            name="gz::sim::systems::MrsMulticopterMotorModel">
            <actuator_number>{index}</actuator_number><robotNamespace>uav</robotNamespace>
            </plugin>"""
            for index in range(4)
        )
        root = ET.fromstring(f"<sdf><model>{plugins}</model></sdf>")
        _normalize_mrs_motor_plugins(root)
        converted = root.findall(".//plugin")
        self.assertEqual(len(converted), 4)
        self.assertTrue(
            all(plugin.attrib["filename"] == "gz-sim-multicopter-motor-model-system" for plugin in converted)
        )
        self.assertEqual(
            [plugin.findtext("motorNumber") for plugin in converted],
            ["0", "1", "2", "3"],
        )
        self.assertTrue(all(plugin.find("robotNamespace") is None for plugin in converted))


if __name__ == "__main__":
    unittest.main()
