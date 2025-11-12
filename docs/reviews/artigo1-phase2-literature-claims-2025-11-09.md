# Phase 2: Literature Claims Validation
**Data:** 2025-11-09
**Status:** In Progress
**Paper:** artigo1 (Comparative Analysis ML Algorithms IoT)

---

## 🎯 OBJECTIVE

Validate all technical claims in the paper that require literature support or verification.

---

## 📋 CLAIMS REQUIRING VALIDATION

### ❌ CLAIM 1: FPR Threshold 1-2%
**Location:** Line 71
**Text:**
> "Operational constraints typically require false positive rates below 1-2% to maintain SOC analyst effectiveness and avoid alert fatigue"

**Status:** ❌ **NO CITATION PROVIDED**
**Severity:** HIGH - Specific numerical claim without source
**Required Action:**
- [ ] Find academic sources for 1-2% FPR threshold
- [ ] Search for SOC alert fatigue literature
- [ ] Alternative: Cite industry reports/whitepapers if academic sources unavailable
- [ ] If no source found: Rephrase as "typically require low false positive rates" without specific number

**Search Strategy:**
- Keywords: "SOC alert fatigue threshold", "false positive rate acceptable", "IDS false alarm rate operational"
- Databases: IEEE, ACM, Google Scholar
- Potential sources: SANS Institute, Gartner reports, SOC operational studies

---

### ❌ CLAIM 2: Lateral Movement (Terminology)
**Location:** Line 71
**Text:**
> "enable adversaries to compromise devices, exfiltrate data, or establish persistent footholds for lateral movement"

**Status:** ⚠️ **TERM USED WITHOUT DEFINITION**
**Severity:** MINOR - Common cybersecurity term but should be cited
**Required Action:**
- [ ] Add brief definition in text OR
- [ ] Cite authoritative source (MITRE ATT&CK, NIST)
- [ ] User question: "What are lateral movements?" → Need to explain

**Proposed Solution:**
Add citation to MITRE ATT&CK framework:
```latex
establish persistent footholds for lateral movement \cite{mitreattack} (the technique of moving from one compromised system to others within the network)
```

---

### ❌ CLAIM 3: Fog/Edge Hardware Specifications
**Location:** Line 139
**Text:**
> "This configuration represents typical fog node or edge server capabilities in IoT deployments."

**Context:** 8-core CPU, 31 GB RAM
**Status:** ❌ **NO CITATION PROVIDED**
**Severity:** HIGH - Specific hardware claim without verification
**Required Action:**
- [ ] Find literature defining fog node hardware specifications
- [ ] Find edge server hardware surveys/benchmarks
- [ ] Verify if 8-core/31GB is indeed "typical"
- [ ] Alternative: Rephrase to "representative of mid-range fog node capabilities" with citation

**Search Strategy:**
- Keywords: "fog computing hardware specifications", "edge server IoT", "fog node resources survey"
- Potential sources: OpenFog Consortium, IEEE fog computing surveys, edge computing benchmarks

---

### ❌ CLAIM 4: IoT Gateway Throughput (1000 packets/sec)
**Location:** Line 312
**Text:**
> "For context, at 1,000 packets/second (typical IoT gateway)"

**Status:** ❌ **NO CITATION PROVIDED**
**Severity:** HIGH - Specific throughput claim without source
**Required Action:**
- [ ] Find IoT gateway throughput specifications
- [ ] Search for IoT network traffic characterization studies
- [ ] Verify 1000 pps is indeed "typical"
- [ ] Alternative: Cite specific gateway product specs or remove "typical"

**Search Strategy:**
- Keywords: "IoT gateway throughput", "IoT network traffic rate", "packets per second IoT"
- Potential sources: IoT gateway vendor specs, IoT traffic characterization papers

---

### ❌ CLAIM 5: Edge Device RAM (512MB-2GB)
**Location:** Lines 310, 344, 350
**Text:**
> "512MB–2GB RAM typical in IoT deployments"
> "Edge devices with limited computational capacity (typically 512MB–2GB RAM, 1-4 cores)"

**Status:** ❌ **NO CITATION PROVIDED**
**Severity:** HIGH - Repeated specific claim without source
**Required Action:**
- [ ] Find edge device hardware surveys
- [ ] Verify RAM ranges for edge computing
- [ ] Search IoT device capability studies
- [ ] Cite product specifications or surveys

**Search Strategy:**
- Keywords: "edge device specifications", "IoT edge computing hardware", "edge device memory constraints"
- Potential sources: Edge computing surveys, IoT device capability studies

---

### ⚠️ CLAIM 6: First Comprehensive CICIoT2023 Baseline
**Location:** Line 336
**Text:**
> "To our knowledge, this study establishes the first comprehensive ML baseline on the CICIoT2023 dataset with rigorous experimental methodology."

**Status:** ⚠️ **NEEDS VERIFICATION**
**Severity:** MEDIUM - "First" claim requires thorough literature review
**Required Action:**
- [ ] Search Google Scholar for "CICIoT2023"
- [ ] Search IEEE Xplore for CICIoT2023 papers
- [ ] Check dataset's official page for citing papers
- [ ] Search arXiv for recent submissions
- [ ] If other studies found: Rephrase claim or highlight differentiators

**Search Strategy:**
```
Google Scholar: "CICIoT2023" OR "CICIOT2023" OR "CIC-IoT-2023"
IEEE Xplore: CICIoT2023
ACM Digital Library: CICIoT2023
arXiv: CICIoT2023 machine learning
```

**Date Range:** 2023-present (dataset published 2023)

---

### ✅ CLAIM 7: Class Imbalance (97.7% attacks)
**Location:** Multiple (lines 118, 342)
**Text:**
> "97.7% attacks, 2.3% benign"

**Status:** ✅ **VALID - Dataset Characteristic**
**Severity:** N/A
**Verification:** This is a factual characteristic of CICIoT2023 dataset, verified in code and data
**No Action Required**

---

### ⚠️ CLAIM 8: Attack Dominance in Real IoT Traffic
**Location:** Line 101
**Text:**
> "class imbalance remains critical—real IoT traffic exhibits attack dominance (often >95%)"

**Status:** ⚠️ **NEEDS CITATION**
**Severity:** MEDIUM - Generalization about real IoT traffic
**Required Action:**
- [ ] Find studies on real-world IoT traffic composition
- [ ] Verify if >95% attack ratio is realistic in production
- [ ] Consider if this refers to datasets vs. real deployments
- [ ] Rephrase or cite sources

**Note:** User mentioned "Verify IoT traffic patterns (attack dominance claim)" - this may be questionable

---

## 📊 SUMMARY

**Total Claims Identified:** 8
**Missing Citations:** 6 (Claims 1-6)
**Needs Verification:** 2 (Claims 6, 8)
**Valid/Verified:** 1 (Claim 7)

---

## 🔍 PRIORITY ORDER

1. **HIGH PRIORITY** (Specific numerical claims without citation):
   - ❌ Claim 1: FPR 1-2% threshold
   - ❌ Claim 3: Fog/edge hardware (8-core/31GB)
   - ❌ Claim 4: 1000 pps IoT gateway
   - ❌ Claim 5: 512MB-2GB edge device RAM

2. **MEDIUM PRIORITY** (Requires literature search):
   - ⚠️ Claim 6: "First comprehensive baseline"
   - ⚠️ Claim 8: Attack dominance in real traffic

3. **LOW PRIORITY** (Easy fixes):
   - ❌ Claim 2: Lateral movement definition

---

## 📝 NEXT STEPS

### Phase 2.1: FPR Threshold Search
- Search academic databases for SOC alert fatigue studies
- Check SANS, Gartner, Forrester reports
- If not found: Rephrase without specific numbers

### Phase 2.2: Hardware Specifications Search
- OpenFog Consortium documentation
- IEEE fog/edge computing surveys
- IoT gateway vendor specifications
- Edge device capability studies

### Phase 2.3: CICIoT2023 Literature Review
- Comprehensive search across all databases
- Check dataset citation page
- Verify "first baseline" claim validity

### Phase 2.4: Traffic Pattern Validation
- Search for real-world IoT traffic studies
- Verify attack dominance claims
- Distinguish datasets vs. production traffic

---

**Status:** Ready to begin systematic literature search
**Estimated Time:** 2-3 hours for all searches
**Tools Needed:** Access to IEEE, ACM, Google Scholar, Zotero
