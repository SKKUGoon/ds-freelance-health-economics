# 041 - Sync Papers With 014 Skill Interpretation

## Objective
Update the English, Korean, and Japanese paper summaries with the new `014_skill` findings and add a repository rule that multilingual paper files must be kept synchronized.

## Scope
- Add `014_skill` results and interpretation to `notebooks/PAPER_ENG.md`, `notebooks/PAPER_KOR.md`, and `notebooks/PAPER_JPN.md`.
- Explain the HANA vs Epurun skill contrast, including the interpretation that Epurun appears more persistent and therefore more favorable to a naive last-value baseline.
- Update `AGENTS.md` with an explicit synchronization rule for the multilingual paper files.

## Files
- `notebooks/PAPER_ENG.md`
- `notebooks/PAPER_KOR.md`
- `notebooks/PAPER_JPN.md`
- `AGENTS.md`

## Implementation Steps
1. Add `014_skill` artifacts to the paper source lists and artifact lists.
2. Insert a new synchronized discussion section in all three papers summarizing bucket-level and per-drug skill findings for HANA and Epurun.
3. Update executive summary, cross-hospital interpretation, limitations, and conclusion text where needed so the `014_skill` interpretation is reflected consistently.
4. Add an `AGENTS.md` rule requiring synchronized updates across the three multilingual paper files unless the user explicitly says otherwise.

## Validation Criteria
- All three paper files contain the same substantive additions in English, Korean, and Japanese respectively.
- The new text cites the `014_skill` findings accurately and does not overstate the Epurun interpretation as proven fact.
- `AGENTS.md` explicitly instructs future agents to keep the three paper files synchronized.
- Markdown structure remains valid and readable.
