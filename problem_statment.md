
Challenge CH-17 — Shipment Frequency Optimization

Domain: Logistics / Shipment Planning

## 1) Business Context
Daily shipping is costly; waiting for full trucks delays stock and frustrates customers.

Why it matters: Recommend optimal shipment frequency per branch based on demand and
cost-service trade-offs.

## 2) Problem Statement
Build a ZIP-based AI solution that solves this challenge end-to-end. Your ZIP package must read the
official competition inputs (to be released later via a Data & Schema Addendum) and generate the
required outputs in the official submission format.

What the solution must do (indicative):

- Optimize shipment frequency to balance cost vs stock availability.
- Decide when to ship (schedule) and how much, given capacity and lead times.
- Minimize delays and stockouts while controlling transportation spend.

Out of scope:

- Using external datasets or calling external APIs at runtime unless explicitly allowed.
- Any manual intervention during evaluation runs.
- Requiring internet access during runtime (runtime is offline).

## 3) Objectives
Primary Objective (Must Have):

- Deliver the required outputs with high task performance and strict schema compliance for
  Shipment Frequency Optimization.

Secondary Objectives (Nice to Have):

- Shipment schedule: shipment dates and quantities per lane/branch.
- Optional: KPI breakdown (cost, fill rate, average delay).

Suggested tools/approaches (non-binding):

- Route and frequency optimisation engines, transport planning software, and simulation
  tools

## 4) Scope and Assumptions
In scope:

- A runnable ZIP package for batch inference.
- End-to-end pipeline: load inputs — process/compute — generate outputs.
- Optimize shipment frequency to balance cost vs stock availability.
- Decide when to ship (schedule) and how much, given capacity and lead times.

Out of scope (unless stated otherwise):

- Human-in-the-loop steps during evaluation.
- Use of external/private datasets.

## 5) Inputs & Data (Data & Schema Addendum)
The official dataset, input file list, and schemas will be released in a separate “Data & Schema
Addendum”. Teams must implement strictly against the published schemas and must not assume
additional fields.

Expected data components (indicative):

- Orders/demand requests with quantities and required dates.
- Capacity, inventory, and lead-time constraints (where applicable).
- Transportation lanes/costs and service-level rules.

Placeholders (will be filled upon release):

- Input files and locations: /data/orders.csv; /data/capacity.csv; /data/transport_costs.csv;
  optional /data/inventory.csv
- Input schema reference (file/link): Defined in this document (assumed) + final schema in
  Addendum.
- Data dictionary reference: Embedded (assumed) + final dictionary in Addendum.
- Train/Test split policy: N/A (optimization) — scenario-based evaluation (details in
  Addendum).
- Allowed external data (if any): Not allowed unless explicitly stated.

## 6) Required Output (Submission Format)
Business meaning of the output:

- Shipment schedule: shipment dates and quantities per lane/branch.; Optional: KPI
  breakdown (cost, fill rate, average delay).
- Output will be validated strictly against the official output schema.

+20 12 22693446    info@coach-academy.net    coach-academy.net    78b, Manial st. Giza, Egypt

Coach
Academy

Indicative output fields/components:

- Shipment schedule: shipment dates and quantities per lane/branch.
- Optional: KPI breakdown (cost, fill rate, average delay).

Placeholders (will be filled upon release):

- Output file name: plan.csv
- Output schema reference (file/link): Embedded (assumed) + final schema in Addendum.
- Uniqueness key (no duplicates): order_id (or order_id + decision_date if required).
- Units, rounding rules, and constraints: allocated_qty as int/float; constraints must be
  respected; costs in float.

## 7) Input/Output Examples
The addendum will publish sample inputs and outputs (synthetic or masked). Below are illustrative
examples only to clarify expected semantics.

Illustrative Sample Input:

inventory.csv: location_id, sku_id, on_hand  
costs.csv: lane_id, fixed_cost, capacity  
[TBD]

Illustrative Sample Output:

shipments.csv: lane_id, ship_date, sku_id, qty

## 8) Functional Requirements
- Run fully inside ZIP in batch mode.
- Read inputs from /data and write the required output file(s) to /output.
- Validate required inputs and fail fast with clear logs if invalid.
- Do not require internet access at runtime; do not call external services at runtime.
- Handle common edge cases gracefully (missing values, unseen IDs, empty inputs), as
  defined in the addendum.

Challenge-specific functional requirements (indicative):

- Respect truck capacity; do not exceed lane constraints.
- Support partial shipments and batching rules as defined.
- Output must be time-consistent (no shipping before inventory is available).

Clarifications teams should anticipate (informative):

- Are current transport contracts flexible enough?
- What is the acceptable lead time per channel?

## 9) Non-Functional Requirements (Prototype + Production Bonus)
Mandatory gates (must pass):

- Produces a valid output file matching the schema.
- Completes within the runtime/memory limits (to be published).
- Does not leak sensitive information via logs or outputs.

Production Bonus expectations (scored):

- Robustness: strong schema validation and graceful error handling.
- Reproducibility: pinned dependencies and deterministic behavior where applicable.
- Operability: clear logs and clear README instructions.
- Efficiency: lower latency and memory usage within constraints.

## 10) ZIP Submission Requirements (Standard)
- Submission is evaluated using a ZIP package + view-only Google Colab link (no
  Docker/container required).
- main.ipynb must run top-to-bottom without manual edits; use relative paths.
- Read official inputs only from /data inside the ZIP; do not assume extra fields beyond the
  schema.
- Write all final artifacts strictly to /output inside the ZIP using the official file names/schema.
- Include clear logs and basic validation checks (required columns, row coverage, no missing
  IDs).

## 11) Deliverables Required from Teams (Concise)
Submit both:

- Google Colab link (shared as Anyone with the link can view)
- One ZIP file with this exact structure:

YourName_ProjectName.zip

main.ipynb

README.txt

technical_report.pdf

/data

/output

README.txt (max 1 page): run steps + expected runtime + hardware assumptions.  
/output: final generated outputs (official names/schema).

technical_report.pdf (max 2-3 pages): approach + design choices + limitations + failure
modes.

## 12) Evaluation Criteria
Total Score = 100 = Solution Score (85%) + Production Bonus (15%).

Solution Score (85%):

- Primary metric (intended): Total logistics cost + stockout penalty (lower is better).
- Secondary: on-time delivery /fill rate and average inventory days.
- Penalty for violating capacity or delivery windows.

Production Bonus (15%):

- Robustness (6%)
- Reproducibility (4%)
- Operability (3%)
- Efficiency (2%)

Tie-breakers:

Better primary solution metric — higher feasibility/safety (if applicable) — lower latency — lower
memory usage.
