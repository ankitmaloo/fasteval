# SOP: Handling and Storage of ESD-Sensitive Items (Warehouse)
**Document ID:** SOP-WH-ESD-001  
**Revision:** 1.0  
**Effective Date:** 2026-02-23  
**Owner:** Warehouse Manager  
**Approved By:** Operations / Quality Manager  
**Applies To:** All warehouse personnel and any visitors entering ESD Protected Areas (EPAs)

---

## 1. Purpose
To prevent Electrostatic Discharge (ESD) damage and related quality failures by standardizing how ESD-sensitive electronic components and assemblies are **received, stored, handled, kitted, issued, returned, and transported** within the warehouse.

This SOP supports product acceptability expectations consistent with **IPC-A-610G, *Acceptability of Electronic Assemblies*** by reducing preventable handling/ESD damage that can lead to latent defects, intermittent failures, or immediate nonconformance.

---

## 2. Scope
This procedure applies to:
- Discrete electronic components (ICs, MOSFETs, diodes, sensors, etc.)
- Printed circuit boards (PCBs) and printed circuit assemblies (PCAs)
- Subassemblies, cables with ESD-sensitive terminations, and electronics modules
- Any item labeled **ESD**, **ESDS**, **Static Sensitive**, or shipped in ESD protective packaging

**Out of scope:** Battery safety, hazmat storage, and temperature-controlled storage (covered in separate SOPs). If an item has both ESD and environmental requirements, **follow both SOPs**.

---

## 3. References (Informational)
- **IPC-A-610G**: *Acceptability of Electronic Assemblies* (handling care expectations, prevention of damage to assemblies, workmanship/acceptability context)
- ANSI/ESD program standards (e.g., **ANSI/ESD S20.20**) and company ESD control plan (if available)
- Manufacturer datasheets/handling notes (MSL, moisture barrier bags, etc.)

> Note: IPC-A-610G is used here as a governing acceptability reference; this SOP does not reproduce IPC text.

---

## 4. Definitions
- **ESD:** Electrostatic discharge—rapid transfer of static charge that can damage electronics.
- **ESDS:** ESD Sensitive (device/item).
- **EPA:** ESD Protected Area—controlled environment where ESD controls are in place.
- **ESD Protective Packaging:** Packaging designed to shield/dissipate static (e.g., shielding bags, conductive totes).
- **Dissipative/Conductive Worksurface:** Benches, mats, shelves designed to safely drain charge.
- **Wrist Strap / Heel Grounder:** Personal grounding devices for operators.
- **Ionizer:** Neutralizes charges on insulators/non-groundable items.

---

## 5. Roles and Responsibilities
**Warehouse Manager**
- Ensures training, compliance, and availability of ESD supplies/equipment.
- Ensures EPA layout is maintained and audited.

**Quality (or ESD Coordinator)**
- Defines ESD control requirements, audit plan, and corrective actions.
- Maintains calibration/verification records for ESD testers/monitors.

**Warehouse Team Members**
- Follow this SOP for all ESDS items.
- Stop work and report any packaging damage, missing labels, or suspected ESD event.

**Visitors/Contractors**
- Must be escorted and comply with EPA entry requirements.

---

## 6. Safety and General Rules
1. **Treat all electronics as ESDS unless clearly marked otherwise.**
2. **No “bare-hand carry” of PCBs/PCAs** in non-ESD areas. Use ESD packaging or totes.
3. **No food/drinks** in EPAs.
4. **No unnecessary plastic/Styrofoam** near ESDS items (high static generators).
5. **Report immediately**: torn shielding bags, missing ESD labels, or dropped/impacted electronics.

---

## 7. Required Equipment and Materials (Minimum)
- ESD wrist straps and/or ESD footwear/heel grounders
- Wrist strap tester and/or footwear tester at EPA entry
- ESD bench mat(s) with ground point(s) (where handling occurs)
- Ground cords and verified ground connections
- ESD shielding bags (metal-in), dissipative bags, conductive totes as appropriate
- ESD labels: “ESD Sensitive”
- ESD-safe tape (or minimized use; do not tape directly to boards)
- Ionizer(s) where required (recommended for insulators, bag opening areas)
- Humidity/temperature monitoring where required by company plan (recommended)
- ESD-safe shelving/totes in ESDS storage locations (preferred)

---

## 8. Establishing and Maintaining ESD Protected Areas (EPAs)
### 8.1 EPA Locations
At minimum, designate:
- **ESDS Receiving/Inspection Station**
- **ESDS Storage Area**
- **Kitting/Issuing Station** (if items are opened/repacked)

All EPAs must have:
- EPA boundary markings/signage
- Defined ground points
- Required test equipment at entry (or at point-of-use)

### 8.2 EPA Entry Procedure (All Personnel)
1. **Remove/secure** static-generating outerwear if feasible (e.g., certain synthetics).
2. Don required PPE:
   - Wrist strap **or** ESD footwear/heel grounders (per company plan).
3. **Test** wrist strap/footwear at tester:
   - If **PASS** → proceed.
   - If **FAIL** → replace device and retest. If still failing, notify supervisor/Quality.
4. Connect wrist strap to approved ground point before touching ESDS items.

**Recordkeeping:** Tester logs are kept per Section 15.

---

## 9. Handling Rules for ESDS Items (Core Procedure)
### 9.1 What You May Touch
- Hold PCBs/PCAs by **edges** only.
- Avoid touching:
  - Component leads, pads, connectors, solder joints, contact surfaces, gold fingers
  - Conformal coating surfaces (if present), optical surfaces, exposed thermal interfaces

### 9.2 Where You May Handle ESDS Items
Only handle **unpackaged** ESDS items:
- On a grounded ESD mat/worksurface **inside an EPA**
- While properly grounded (wrist strap/footwear) and after passing tester

### 9.3 Prohibited Actions
- Do not place boards on:
  - Cardboard, plain plastic, bubble wrap, foam (unless ESD-rated), clothing, or bare metal
- Do not slide PCBs across surfaces (reduces tribocharging and mechanical abrasion)
- Do not stack boards without approved separators/totes
- Do not use compressed air that is not ESD-safe/ionized near exposed electronics

### 9.4 ESD Event / Suspected Damage Response
If any of the following occurs: spark felt/heard, dropped board, torn shielding bag, item found unpackaged in non-EPA:
1. **Stop** and isolate item in ESD shielding bag/tote.
2. Label as **HOLD – SUSPECT ESD/HANDLING EVENT**.
3. Notify Quality and supervisor for disposition (inspection/testing as required).

---

## 10. Receiving Procedure (ESDS Shipments)
1. Identify ESDS indicators:
   - ESD caution symbols, shielding packaging, manufacturer labels
2. Move shipment to **ESDS Receiving Station (EPA)** before opening inner packaging.
3. Verify receiver is grounded and passed strap/footwear test.
4. Inspect packaging condition:
   - If shielding bag is torn/punctured or seals compromised → place item on **HOLD** and notify Quality.
5. Verify labeling and traceability (as required by your system):
   - Part number, revision, lot/date code, quantity, supplier, PO
6. If repack is necessary:
   - Use **ESD shielding bags or conductive totes**
   - Add ESD label if not already present
7. Update inventory location to ESDS storage.

---

## 11. Storage Procedure (ESDS Items)
### 11.1 Storage Location Requirements
- Store ESDS items only in designated **ESDS storage zones**.
- Preferred: ESD-safe shelving/totes. At minimum:
  - Items remain **inside shielding packaging** unless actively handled in an EPA.
- Keep storage clean, dry, and free of loose plastics/Styrofoam.

### 11.2 Storage Configuration
- Keep items in original manufacturer packaging when possible.
- Do not overload bins; prevent bending/warping of PCBs.
- Use dividers or individual slots for PCBs/PCAs.
- Heavy items **not on top** of electronics.

### 11.3 Labeling
Each stored ESDS container must have:
- ESD label (if not inherently obvious)
- Part number and quantity
- Lot/date code (if applicable)
- Status label if on hold (HOLD / QUARANTINE / RETURN)

---

## 12. Internal Transport (Within Warehouse)
1. For any movement outside an EPA, ESDS items must be:
   - In a **closed ESD shielding bag** (sealed/closed) **or**
   - In a **covered conductive tote** with ESD label
2. Do not carry uncovered PCBs in hands through aisles.
3. Place totes on carts that are clean and (preferred) ESD-safe.
4. Avoid friction/rapid movement that increases charge generation.

---

## 13. Kitting and Issuing to Production / Technicians
1. Perform kitting only in an **EPA**.
2. Ground operator and verify PASS on tester.
3. Open packaging only at ESD workstation.
4. Verify correct part, revision, quantity, and any special requirements (e.g., moisture controls).
5. Package kit for transfer:
   - Use shielding bag/tote; include ESD label
   - Include traveler/pick list in a document pouch (avoid loose paper rubbing on boards)
6. Issue items using standard inventory transaction steps.

**Key requirement:** Items should arrive to the user still in ESD protective packaging until point-of-use.

---

## 14. Returns to Stock (RTS) and Unused Materials
1. Returned ESDS items must be received into an **EPA**.
2. Inspect for:
   - Opened/torn packaging
   - Missing labels/traceability
   - Bent pins, contamination, scratches, cracked components, connector damage
3. If packaging is damaged or traceability uncertain:
   - Place on **HOLD** for Quality disposition.
4. Repack using proper shielding packaging and labels.
5. Return to ESDS storage location with correct inventory transaction.

---

## 15. Verification, Audits, and Records
### 15.1 Daily/Shift Checks (Warehouse Lead or Designee)
- Verify EPA signage/boundaries intact
- Verify wrist strap/footwear tester operational
- Verify ESD mats and ground cords connected and not damaged
- Verify availability of shielding bags/totes and ESD labels

### 15.2 Periodic Audits (Quality/ESD Coordinator)
- Audit compliance with:
  - Grounding practices (PASS test usage)
  - Proper packaging during transport/storage
  - Housekeeping and prohibited materials
- Verify/measure grounding points and ESD surfaces per company plan.

### 15.3 Records to Maintain
- Training completion records (Section 16)
- ESD audit results and corrective actions
- Tester verification/calibration records (as applicable)
- Incident/HOLD logs for suspected ESD/handling events

---

## 16. Training Requirements
All warehouse personnel handling ESDS items must complete:
- Initial ESD awareness and handling training (before independent work)
- Annual refresher training (or per management requirement)
- On-the-job demonstration of:
  - Proper grounding and tester use
  - Correct packaging selection and sealing/labeling
  - Proper PCB/PCA handling (edges only)

Training should reinforce that preventing handling damage supports acceptability expectations consistent with **IPC-A-610G** (e.g., avoiding physical damage/contamination that can render assemblies unacceptable).

---

## 17. Quick Reference: Do / Don’t
### DO
- Do test wrist strap/footwear and get **PASS** before handling
- Do handle boards by **edges**
- Do keep items in **shielding packaging** outside the EPA
- Do use conductive totes for internal moves
- Do label and quarantine anything suspicious

### DON’T
- Don’t touch leads, connectors, or solder joints
- Don’t place electronics on regular plastic, foam, or cardboard
- Don’t transport uncovered boards through aisles
- Don’t ignore torn bags or missing ESD labels
- Don’t “assume it’s fine” after a drop or ESD incident—**report it**

---

## 18. Appendix A — Packaging Selection Guide (Warehouse Use)
| Situation | Minimum Requirement | Preferred |
|---|---|---|
| Long-term storage | Shielding bag or conductive tote | Original manufacturer packaging inside shielding bag + labeled tote |
| Move outside EPA | Closed shielding bag OR covered conductive tote | Conductive tote with lid + ESD label |
| Kitting multiple items | Shielding bags per item or separators | Conductive tote with dividers + ESD document pouch |
| Returns to stock | Re-shield and relabel | New shielding bag if original compromised |

---

## 19. Appendix B — HOLD Tag Triggers
Apply **HOLD – SUSPECT ESD/HANDLING EVENT** when:
- Shielding bag torn/punctured/open with unknown history
- Item found unpackaged outside EPA
- Drop/impact occurred
- Missing traceability label/lot control data (when required)
- Evidence of bent pins, cracked parts, deep scratches, contamination, corrosion

---

**End of SOP**