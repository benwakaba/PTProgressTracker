"""Published numbers the simulator is calibrated to or checked against.

Every number here was read from the paper's abstract (PubMed). Nothing else in
the study is taken from real data. The split between "calibration" and
"validation" matters: calibration targets are hit by construction (calibrate()
in simulate.py solves for them), so agreement with them is not evidence of
anything. Only the validation targets are checks.
"""

WALLWORK_2024 = {
    "citation": (
        "Wallwork SB, Braithwaite FA, O'Keeffe M, et al. The clinical course of acute, "
        "subacute and persistent low back pain: a systematic review and meta-analysis. "
        "CMAJ. 2024;196(2):E29-E46. doi:10.1503/cmaj.230542"
    ),
    # Pooled mean pain (0-100) with 95% CI, by weeks since study entry.
    "acute": {0: (56, 49, 62), 6: (26, 21, 31), 26: (22, 18, 26), 52: (21, 17, 25)},
    "persistent": {0: (56, 37, 74), 6: (48, 32, 64), 26: (43, 29, 57), 52: (40, 27, 54)},
}

ARTUS_2010 = {
    "citation": (
        "Artus M, van der Windt DA, Jordan KP, Hay EM. Low back pain symptoms show a similar "
        "pattern of improvement following a wide range of primary care treatments: a "
        "systematic review of randomized clinical trials. Rheumatology (Oxford). "
        "2010;49(12):2346-2356. doi:10.1093/rheumatology/keq245"
    ),
    # Pooled within-group standardized mean improvement in pain (118 RCTs), by week.
    "pain_smd": {6: (0.86, 0.65, 1.07), 13: (1.07, 0.87, 1.27), 27: (1.03, 0.82, 1.25), 52: (0.88, 0.60, 1.10)},
}

FARRAR_2001 = (
    "Farrar JT, Young JP Jr, LaMoreaux L, Werth JL, Poole RM. Clinical importance of changes in "
    "chronic pain intensity measured on an 11-point numerical pain rating scale. Pain. "
    "2001;94(2):149-158. doi:10.1016/S0304-3959(01)00349-9"
)
FRANSEN_2015 = (
    "Fransen M, McConnell S, Harmer AR, et al. Exercise for osteoarthritis of the knee. Cochrane "
    "Database Syst Rev. 2015;1:CD004376. doi:10.1002/14651858.CD004376.pub3"
)
HAYDEN_2021 = (
    "Hayden JA, Ellis J, Ogilvie R, Malmivaara A, van Tulder MW. Exercise therapy for chronic low "
    "back pain. Cochrane Database Syst Rev. 2021;9:CD009790. doi:10.1002/14651858.CD009790.pub2"
)
ARGENT_2018 = (
    "Argent R, Daly A, Caulfield B. Patient involvement with home-based exercise programs: can "
    "connected health interventions influence adherence? JMIR Mhealth Uhealth. 2018;6(3):e47. "
    "doi:10.2196/mhealth.8518"
)
ESSERY_2017 = (
    "Essery R, Geraghty AW, Kirby S, Yardley L. Predictors of adherence to home-based physical "
    "therapies: a systematic review. Disabil Rehabil. 2017;39(6):519-534. "
    "doi:10.3109/09638288.2016.1153160"
)
JACK_2010 = (
    "Jack K, McLean SM, Moffett JK, Gardiner E. Barriers to treatment adherence in physiotherapy "
    "outpatient clinics: a systematic review. Man Ther. 2010;15(3):220-228. "
    "doi:10.1016/j.math.2009.12.004"
)
PISTERS_2010 = (
    "Pisters MF, Veenhof C, Schellevis FG, et al. Exercise adherence improving long-term patient "
    "outcome in patients with osteoarthritis of the hip and/or knee. Arthritis Care Res. "
    "2010;62(8):1087-1094. doi:10.1002/acr.20182"
)

# --- Calibration targets (hit by construction) -------------------------------

# Mean pain at week 6 as a fraction of baseline (Wallwork 2024).
SIX_WEEK_RATIO_ACUTE = 26 / 56
SIX_WEEK_RATIO_PERSISTENT = 48 / 56

# Exercise vs. control difference in pain at the end of treatment, 0-10 scale:
# 12/100 for knee OA (Fransen 2015) and 15.2/100 for chronic LBP (Hayden 2021).
EXERCISE_EFFECT_POINTS = (1.2 + 1.52) / 2

# Non-adherence to home exercise is reported "as high as 50%" (Argent 2018) and
# "can reach 70%" (Essery 2017), with no shared definition. Operationalized as
# the share of patients who exercise on fewer than half of plan days.
SHARE_LOW_ADHERENCE = 0.50

# --- Definitions ---------------------------------------------------------------

# ~30% reduction on the 0-10 NRS is a clinically important improvement (Farrar 2001).
RESPONDER_REDUCTION = 0.30

# --- Validation targets (not used for tuning) ---------------------------------


def wallwork_ratio(cohort: str, week: int) -> tuple:
    """Mean pain at `week` over baseline, with the CI scaled by the baseline point estimate."""
    base = WALLWORK_2024[cohort][0][0]
    mean, lo, hi = WALLWORK_2024[cohort][week]
    return mean / base, lo / base, hi / base


def artus_6_over_13() -> tuple:
    """Share of the 13-week improvement already present at week 6.

    The interval is the crude ratio of CI endpoints (the covariance between the
    two estimates is not reported), so it is wider than a proper CI.
    """
    m6, lo6, hi6 = ARTUS_2010["pain_smd"][6]
    m13, lo13, hi13 = ARTUS_2010["pain_smd"][13]
    return m6 / m13, lo6 / hi13, hi6 / lo13
