# Dataset Documentation

## Source and Scope

The project uses the Ageing Data of Lithium-ion Panasonic NCR18650B Battery Cells dataset from Zenodo (record 7896429). The complete collection contains 116 battery cells tested under 37 aging profiles and is distributed as a large set of HDF5 files. Each HDF5 file represents one battery cell. The project uses a manageable subset of cells rather than processing the complete collection.

Dataset source: <https://zenodo.org/records/7896429>

## Pilot File

The current pilot file is `2_009.h5`. It contains one cell, 57 test records, and approximately 13.7 million measurement rows. The pilot workflow creates a compact table with one row per test record. These pilot outputs are not final results for the complete dataset.

The pilot quality summary reports:

- 0 non-finite values in the inspected arrays
- 51 temperature sentinel values near `-273.16`
- 4 non-increasing time transitions flagged for review
- 57 rows in the pilot feature table

## Data Dictionary

| Field | Meaning | Unit or Format | Project Use |
|---|---|---|---|
| `cell_id` | Identifier for the battery cell | Identifier | Keeps complete cells separate for unseen-cell testing |
| `idtest` | Test identifier within a cell file | Identifier | Links metadata to the matching HDF5 test group |
| `idexp` / `experiment` | Operating-profile identifier or label | Identifier or text | Describes the tested operating condition |
| `start_datetime` | Start time recorded for a test | Dataset timestamp value | Provides test timing information |
| `schedule_file` | Schedule associated with the test | File or label | Documents the applied test schedule |
| `voltage` | Battery voltage measurements | V | Candidate health-related measurement |
| `current` | Charge or discharge current measurements | A | Describes battery load |
| `temperature` | Battery temperature measurements | °C | Operating condition that affects aging |
| `test_time` | Measurement time within a test | Time value | Used to calculate test duration and check ordering |
| `chargecap` | Charge-capacity measurements within a test | Capacity value | Used when deriving capacity features |
| `dischargecap` | Discharge-capacity measurements within a test | Capacity value | Used to estimate capacity and SOH |
| `stepindex` | Measurement or test-step index | Integer | Identifies steps within a test |
| `soh_proxy_percent` | Preliminary SOH estimate from the pilot capacity estimate | Percent | Temporary pilot target; final SOH definition is still pending |

## Operating-Profile Factors

The aging profiles vary temperature, charge current, average discharge current, peak discharge current, load frequency, average SOC, and SOC swing. These factors are documented because they may change the rate and pattern of battery degradation.

## Data-Handling Notes

The HDF5 test groups must be matched to metadata using the test identifier. The metadata rows are not necessarily stored in the same order as the test-group names, so matching them by sorted position can mislabel tests. Capacity arrays also reset within a test, so capacity calculations must use differences within the appropriate step. Temperature sentinel values and time-order issues should be flagged and handled according to the group’s confirmed preprocessing rules.

The cell identifier and operating-profile information should remain attached to every extracted row. They are needed to create leakage-safe random, unseen-cell, and unseen-profile evaluation splits.

## Current Status

The dataset source, pilot file, main variables, operating-profile factors, and initial quality findings have been documented. The final selected cell list, cycle definition, feature table, and SOH target definition remain subject to confirmation before final modeling.
