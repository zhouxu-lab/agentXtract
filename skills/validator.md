# Dielectric-property validation guidance

Use the numerical limits in `configs/thresholds.yaml` as the authoritative
acceptance bounds. Reject a record only when a configured bound is violated;
flag unusual but permitted patterns for inspection.

## Required checks

- Frequency, temperature, moisture, dielectric constant, and loss factor must
  lie within their configured ranges when present.
- Apply the wider low-frequency dielectric limits at or below the configured
  radio-frequency boundary.
- Apply the loss-tangent ceiling only above that boundary; ionic conduction can
  produce large loss factors at lower frequencies.
- Do not treat a missing optional condition as zero.
- Wet- and dry-basis moisture values are distinct conditions.
- Electrical conductivity and salt concentration are distinct fields.
- Values computed from source equations must retain the equation, fitted
  domain, fit statistic when reported, and `equation_derived` provenance.

## Physical consistency

- Dielectric constant and loss factor must be positive.
- Increasing moisture often increases both properties, but composition,
  density, phase state, and frequency can alter the trend. Do not enforce a
  universal monotonic relationship.
- Ionic conductivity contributes approximately inversely with frequency, so
  moist ionic materials can have much larger loss factors in the
  radio-frequency range than in the microwave range.
- Heating can shift relaxation behavior and increase ionic mobility. Either
  increasing or decreasing temperature trends may be physically plausible.
- Fat and entrained air commonly reduce bulk dielectric response.
- A model-derived point is valid only inside both the model's declared domain
  and the experimental range reported by the source.

## Structural checks

- When a source reports both real and imaginary permittivity at the same
  conditions, they should normally be paired in one record.
- Never merge rows across different source identities, materials, moisture
  bases, salt levels, conductivities, frequencies, or temperatures.
- Preserve genuine replicates. Remove only condition-and-value duplicates.
- Comparison values attributed to another source must not be represented as
  measurements from the current source.
- Log every rejected row and the field that violated a configured bound.

The public prompt intentionally contains no source-specific measurements or
correction rules. Projects that require narrower domain limits should use a
local configuration rather than embedding study records in this file.
