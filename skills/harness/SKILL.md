---
name: harness
description: "Knowledge and operating principles from DeciCAD (Harness-Grounded Learning of Action Policies for CAD Agents). Guides CAD agents to operate at the Semantic Decision Bottleneck: semantic actions, state-grounded execution, spatial contracts, two-tier observation, and structured error recovery."
metadata:
  source: DeciCAD (arXiv:2610.21755v1)
---

# DeciCAD Agent Harness Knowledge & Guidelines

This skill encapsulates the principles from **DeciCAD: Harness-Grounded Learning of Action Policies for CAD Agents** (Peng et al., 2026). It instructs CAD agents on how to model efficiently, reliably, and recoverably by operating at the **Semantic Decision Bottleneck**.

---

## 1. The Harness-Induced CAD Decision Space

A CAD agent harness defines the model-facing interface $H = (A_H, \beta_H, O_H)$:
- $A_H$: **Admissible Actions** — Semantic, typed CAD operations (e.g. `SketchAddLines`, `SketchAddArc`, `SolidExtrude`, `ConstraintAdd`) rather than arbitrary programming boilerplate or raw Python scripts.
- $\beta_H$: **State Grounding** — Binds semantic actions into transitions on a persistent CAD document containing stable object references (`Ref<T>`).
- $O_H$: **Decision-Oriented Observations** — Returns structured, compact execution feedback and error diagnosis rather than noisy, token-heavy full document dumps.

### The Semantic Decision Bottleneck (SDB)
- **Action Sufficiency**: The action space must expose every necessary geometric transition needed to reach optimal geometry, without forcing the model to invent arbitrary program flow, helper variables, or code syntax.
- **Observation Sufficiency**: The observation space must distinguish backend states that require incompatible next actions (e.g., distinguishing an overconstrained sketch conflict from a degenerate geometry failure).
- **Compression Frontier**: Compress representation burden (input/output tokens) while strictly preserving decision and recovery capability.

---

## 2. Shared Spatial Contract

All CAD actions and specifications must obey a strict spatial contract:
1. **Units**:
   - Lengths: strictly in millimeters (`mm`).
   - Angles: strictly in degrees (`deg`).
2. **Coordinate Frames**:
   - **World Frame**: Right-handed Cartesian coordinates: $(+X)$ right, $(+Y)$ back/up, $(+Z)$ up.
   - **Sketch Local Frame**: 2D entities exist strictly within the right-handed 2D coordinate system of their sketch plane.
   - **Rotations**: Follow the standard right-hand rule (`axis`, `angle`).
3. **Parametric Scaling**:
   - Normalize parametric proportions (e.g., expressions of $h$) to explicit millimeter values before modeling.

---

## 3. State Grounding & Object References (`Ref<T>`)

1. **Persistent Document State**:
   - Each modeling action modifies the shared persistent CAD document.
   - Never regenerate an entire drawing from scratch when correcting an error; apply targeted incremental actions to the active document.
2. **Stable References**:
   - Objects, sketches, bodies, and geometric entities have stable identifiers.
   - Downstream operations (e.g. constraints, fillets, extrusions) must reference existing stable IDs rather than guessing names.
3. **Closed Contours First**:
   - Profiles intended for extrusion or solid modeling must be continuous, topologically closed loops.
   - Every entity endpoint must exactly coincide with the next entity start point.

---

## 4. Two-Tier Observation Protocol

Instead of repeatedly dumping the entire CAD document state (which causes state aliasing and context inflation), use a two-tier observation protocol:

1. **Passive Execution Feedback (Consequence Evidence)**:
   - Immediately inspect the compact result returned by each tool call:
     - On success: verify affected entities, solver status, and geometric validity.
     - On failure: read the structured error category and recovery hint.
2. **Active Hierarchical Inspection**:
   - When more context is needed, query hierarchically:
     - *Discovery*: List objects, sketch geometries, or constraint collections (`list_objects`, `InspectList`).
     - *Detail Retrieval*: Inspect exact geometry, coordinates, or solver state for a specific target (`inspect_object`, `InspectGet`).

---

## 5. Structured Error Taxonomy & Recovery

When an operation fails, categorize the failure and apply targeted recovery:

| Error Category | Root Cause | Recommended Recovery Action |
| :--- | :--- | :--- |
| `InvalidInput` | Malformed coordinates, invalid radius, or out-of-range parameters | Validate against dimensions; use exact calculator tools (`geometry_tools`). |
| `MissingObject` | Referenced entity ID does not exist or was deleted | Inspect active sketch/document to obtain the valid entity reference. |
| `InvalidState` | Inactive sketch, missing body, or uncomputed document | Recompute document or activate the target sketch before operating. |
| `GeometryFailure` | Self-intersecting profile, open contour, or zero-length line | Check endpoint coincidence; ensure lines and arcs share exact intersection points. |
| `Conflict` / `Overconstrained` | Redundant constraints (e.g. coincident added to tangent joint) | Remove the redundant constraint. Never combine Coincident and Tangent on the same vertex joint! |
| `OutcomeUnknown` | Network timeout or unclear execution status | Call `inspect_object` or `get_screenshot` to verify current state before retrying. |

---

## 6. Exact Geometry Over Manual Approximation

- **Never estimate arc angles or trim by eye**:
  - Tangent corners between lines: use `fillet_two_lines`.
  - Tangents between two circles: use `tangent_lines_two_circles`.
  - Tangents from a point to a circle: use `tangent_from_point_to_circle`.
  - Circle-to-circle intersections: use `circle_circle_intersections`.
  - Lines meeting a circle: use `circle_line_intersections`.
  - Arc directions: use `arc_angles` with a `via_point` on the kept contour side.
- **Constraints**:
  - For a tangent arc-line joint: use **ONE** `Tangent` constraint (which inherently enforces coincidence).
  - Add dimensional (radius, distance) constraints only after geometric topology is fully connected.
