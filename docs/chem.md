# Chemistry

The chemistry stack runs an OT-2 style deck in simulation, checks every
protocol against the human reagent allowlist, and can emit an Opentrons
protocol plus HTTP calls for a real robot. The virtual lab is what the
examples run.

## Allowlist

`configs/chem/reagents.yaml` is human-maintained. No function in physlab
opens it for writing, and an agent must not edit it. The file permits only:

| Id | Limit |
| --- | --- |
| water | 55.5 M, 100 mL |
| food_dye | 0.01 M, epsilon 15000 L mol⁻¹ cm⁻¹ at 520 nm |
| sodium_chloride | 5 M |
| acetic_acid | 1 M, Ka 1.8×10⁻⁵ |
| sodium_hydroxide | 0.1 M |
| phenolphthalein | 0.001 M |

Food dye and sodium hydroxide are incompatible. Acetic acid and sodium
hydroxide are not, so a titration is allowed. Incubation above `max_temp_c`
(40 °C) is denied. In real mode the check also requires an approval token.
Issuing that token belongs to the safety gate, not to this module.

`ChemSafetyPolicy.check_protocol` returns a `SafetyDecision`. Unknown
reagents, over-limit concentration or volume, incompatible mixtures, and
unsupported step kinds are violations.

## Deck and protocol

Slots are `"1"` through `"11"`. Labware kinds are `well_plate_96` (300 µL),
`tube_rack_15ml`, `tube_rack_50ml`, `reservoir_12`, and `tiprack_300ul`.
Wells are `slot:well`, for example `2:A1`.

Steps are `transfer` (with optional `mix_after`), `mix`, `incubate`,
`read_absorbance`, and `read_ph`. Validation rejects volumes that overflow a
well or a transfer that needs more 300 µL tips than the tip racks hold.

## Virtual lab

Mixing is ideal. A transfer moves each solute in proportion to the volume
fraction. pH of a strong base plus a weak acid solves the charge balance

`[H+] + C_base = K_w/[H+] + C_acid K_a / ([H+] + K_a)`

with `scipy.optimize.brentq`. Absorbance is Beer-Lambert using the
per-wavelength epsilon in the allowlist and `path_cm`. First-order A to B
kinetics, with an optional Arrhenius rate, run only for pairs listed in
the allowlist whose reactant and product are both allowlisted. The shipped
file lists no pairs.

The plate reader and pH probe add Gaussian noise from a seeded generator.

`ExperimentRunner.run_chem` stores one trial per reading under
`runs/<run_id>/`, so `fit_models` can fit the table.

## Examples

```bash
uv run python examples/10_dye_calibration_curve.py
uv run python examples/10b_titration_curve.py
```

The dye example fits absorbance against concentration and exits 1 when the
slope is more than 3% from the allowlisted epsilon. The titration example
exits 1 when the steepest pH segment is more than 2% from 1000 µL.

## Verified APIs

Checked against the public docs on 2026-11-02. These are not guessed.

Python Protocol API, <https://docs.opentrons.com/python-api/versioning/>:

- The latest API level in the docs is `2.30` (Flex, robot software 10.0.0).
- OT-2 on the latest software supports `2.0` through `2.28`.
- `apiLevel` goes in `metadata` or in `requirements`, not both.
- `requirements` may set `"robotType": "OT-2"`.
- Generated protocols default to `2.28` and take another level as an argument.

Labware load names used in the generated file, from
<https://docs.opentrons.com/python-api/labware/>,
<https://docs.opentrons.com/python-api/tutorial/>, and
<https://docs.opentrons.com/python-api/examples/>:

- `opentrons_96_tiprack_300ul`
- `corning_96_wellplate_360ul_flat`
- `nest_12_reservoir_15ml`
- `p300_single_gen2` on the left mount
- `pipette.transfer(..., new_tip="always", mix_after=(3, volume))`
- `opentrons_15_tuberack_falcon_15ml_conical` and
  `opentrons_6_tuberack_falcon_50ml_conical` are the OT-2 labware-library
  load names for the 15 mL and 50 mL racks.

HTTP API, <https://docs.opentrons.com/http/api_reference.html>, served on
port 31950:

- Every request sends `Opentrons-Version`. The documented values are an
  integer `2` or higher, or `*` for the latest. The client sends `*`.
- `POST /protocols` uploads a protocol as `multipart/form-data` with the
  file part named `files`. The response id is `data.id`.
- `POST /runs` with JSON `{"data": {"protocolId": "<id>"}}` creates a run.
- `GET /runs/{runId}` returns `data.status`.

Tests call the client only through `httpx.MockTransport`. Nothing in the
suite contacts a robot.
