-- prodrome warehouse schema.
--
-- Three layers, matching the dbt project that reads them:
--   raw_*     immutable landings of what the APIs returned, one row per request
--   (staging and marts live in dbt/models and are views/tables over these)
--
-- Design rules applied throughout:
--   * Every fact table carries `as_of_quarter`, because every number in this
--     project is a point-in-time number. A table without it would be ambiguous.
--   * Every fact table carries `run_id`, so a figure traces to the run, the
--     config and the traffic that produced it.
--   * Counts are BIGINT: the grand total N exceeds 20 million and the product of
--     two margins overflows INTEGER.
--   * No surrogate keys. (drug_unii, reaction, as_of_quarter) is the natural key
--     and making it explicit keeps the point-in-time grain impossible to lose.

CREATE TABLE IF NOT EXISTS runs (
    run_id              VARCHAR PRIMARY KEY,
    started_at          TIMESTAMP NOT NULL,
    finished_at         TIMESTAMP,
    command             VARCHAR NOT NULL,
    prodrome_version    VARCHAR NOT NULL,
    config_digest       VARCHAR NOT NULL,   -- sha256 of the resolved config
    cohort_size         INTEGER NOT NULL,
    first_quarter       VARCHAR NOT NULL,
    last_quarter        VARCHAR NOT NULL,
    openfda_last_updated DATE,
    has_openfda_key     BOOLEAN NOT NULL,
    embed_backend       VARCHAR,
    -- Traffic accounting, so quota use is auditable rather than folklore.
    requests            INTEGER DEFAULT 0,
    cache_hits          INTEGER DEFAULT 0,
    retries             INTEGER DEFAULT 0,
    notes               VARCHAR
);

-- The cohort as it stood for a given run. Snapshotted rather than joined to
-- config, so a historical run's population is recoverable after conf/ changes.
CREATE TABLE IF NOT EXISTS raw_cohort (
    run_id          VARCHAR NOT NULL,
    drug_unii       VARCHAR NOT NULL,
    drug_name       VARCHAR NOT NULL,
    spl_set_id      VARCHAR,
    substance_names VARCHAR,          -- JSON array
    brand_names     VARCHAR,          -- JSON array
    unii_coverage   DOUBLE,
    notes           VARCHAR,
    PRIMARY KEY (run_id, drug_unii)
);

-- Point-in-time 2x2 tables. The central fact table.
CREATE TABLE IF NOT EXISTS raw_contingency (
    run_id          VARCHAR NOT NULL,
    drug_unii       VARCHAR NOT NULL,
    reaction        VARCHAR NOT NULL,
    as_of_quarter   VARCHAR NOT NULL,
    a               BIGINT NOT NULL,
    b               BIGINT NOT NULL,
    c               BIGINT NOT NULL,
    d               BIGINT NOT NULL,
    -- How `a` was obtained: a truncated count aggregation cannot distinguish
    -- "absent" from "unknown", so the provenance is recorded per cell rather
    -- than assumed. See prodrome.clients.openfda.CountResponse.
    a_source        VARCHAR NOT NULL,   -- 'count_exhaustive' | 'count_present' | 'targeted'
    PRIMARY KEY (run_id, drug_unii, reaction, as_of_quarter)
);

-- Reconstructed label timeline: one row per retrievable SPL version.
CREATE TABLE IF NOT EXISTS raw_label_version (
    run_id          VARCHAR NOT NULL,
    drug_unii       VARCHAR NOT NULL,
    spl_set_id      VARCHAR NOT NULL,
    spl_version     INTEGER NOT NULL,
    published_date  DATE,
    effective_date  DATE,
    -- The date prodrome treats the version as taking effect: effective_date when
    -- the document declares one, else the DailyMed publication date.
    authoritative_date DATE NOT NULL,
    core_char_count INTEGER NOT NULL,
    section_count   INTEGER NOT NULL,
    is_usable       BOOLEAN NOT NULL,
    PRIMARY KEY (run_id, drug_unii, spl_version)
);

-- Label-mention verdicts, per reaction per label version.
CREATE TABLE IF NOT EXISTS raw_label_mention (
    run_id              VARCHAR NOT NULL,
    drug_unii           VARCHAR NOT NULL,
    spl_version         INTEGER NOT NULL,
    authoritative_date  DATE NOT NULL,
    reaction            VARCHAR NOT NULL,
    verdict             VARCHAR NOT NULL,
    is_labelled         BOOLEAN NOT NULL,
    best_semantic_score DOUBLE,
    best_semantic_z     DOUBLE,
    evidence_method     VARCHAR,
    evidence_score      DOUBLE,
    evidence_section_code VARCHAR,
    evidence_section_name VARCHAR,
    evidence_tier       VARCHAR,
    evidence_snippet    VARCHAR,
    evidence_via        VARCHAR,
    PRIMARY KEY (run_id, drug_unii, spl_version, reaction)
);

-- Per-quarter fitted GPS prior, one per quarter. Stored because the shrinkage
-- applied to a cell is only interpretable alongside the prior it came from.
CREATE TABLE IF NOT EXISTS raw_gps_prior (
    run_id          VARCHAR NOT NULL,
    as_of_quarter   VARCHAR NOT NULL,
    a1 DOUBLE, b1 DOUBLE, a2 DOUBLE, b2 DOUBLE, p DOUBLE,
    log_likelihood  DOUBLE,
    n_cells         INTEGER,
    converged       BOOLEAN,
    PRIMARY KEY (run_id, as_of_quarter)
);

-- Per-quarter empirical null from negative controls.
CREATE TABLE IF NOT EXISTS raw_empirical_null (
    run_id          VARCHAR NOT NULL,
    as_of_quarter   VARCHAR NOT NULL,
    mu              DOUBLE,
    sd              DOUBLE,
    n_controls      INTEGER,
    converged       BOOLEAN,
    PRIMARY KEY (run_id, as_of_quarter)
);

-- Quarterly new-report counts per pair, for spike and notoriety diagnostics.
CREATE TABLE IF NOT EXISTS raw_quarterly_reports (
    run_id          VARCHAR NOT NULL,
    drug_unii       VARCHAR NOT NULL,
    reaction        VARCHAR NOT NULL,
    quarter         VARCHAR NOT NULL,
    new_reports     BIGINT NOT NULL,
    PRIMARY KEY (run_id, drug_unii, reaction, quarter)
);

-- Reporter-composition breakdowns, for the artefact diagnostics.
CREATE TABLE IF NOT EXISTS raw_reporter_mix (
    run_id          VARCHAR NOT NULL,
    drug_unii       VARCHAR NOT NULL,
    reaction        VARCHAR NOT NULL,
    as_of_quarter   VARCHAR NOT NULL,
    dimension       VARCHAR NOT NULL,   -- 'country' | 'qualification'
    category        VARCHAR NOT NULL,
    reports         BIGINT NOT NULL,
    PRIMARY KEY (run_id, drug_unii, reaction, as_of_quarter, dimension, category)
);

CREATE INDEX IF NOT EXISTS idx_contingency_pair
    ON raw_contingency (drug_unii, reaction);
CREATE INDEX IF NOT EXISTS idx_contingency_quarter
    ON raw_contingency (as_of_quarter);
CREATE INDEX IF NOT EXISTS idx_mention_pair
    ON raw_label_mention (drug_unii, reaction);

-- ---------------------------------------------------------------------------
-- Analysis outputs.
--
-- These are computed in Python rather than dbt because they need scipy: the
-- gamma-Poisson MLE, the gamma posterior quantiles behind EB05, and the
-- empirical-null fit have no SQL expression. dbt then builds staging and mart
-- models over both these and the raw_* tables -- the division is "SQL where SQL
-- is honest, Python where the statistics live", not one tool used for everything.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS stat_disproportionality (
    run_id          VARCHAR NOT NULL,
    drug_unii       VARCHAR NOT NULL,
    reaction        VARCHAR NOT NULL,
    as_of_quarter   VARCHAR NOT NULL,
    a BIGINT, b BIGINT, c BIGINT, d BIGINT, n BIGINT,
    expected        DOUBLE,
    prr             DOUBLE,
    prr_ci_lower    DOUBLE,
    prr_ci_upper    DOUBLE,
    ror             DOUBLE,
    ror_ci_lower    DOUBLE,
    ror_ci_upper    DOUBLE,
    chi2_yates      DOUBLE,
    rrr             DOUBLE,
    log2_oe_shrunk  DOUBLE,
    log2_oe_ci_lower DOUBLE,
    log2_oe_ci_upper DOUBLE,
    ebgm            DOUBLE,
    eb05            DOUBLE,
    eb95            DOUBLE,
    gps_posterior_noise_weight DOUBLE,
    -- Empirical calibration against the negative controls for this quarter.
    calibrated_p    DOUBLE,
    raw_p           DOUBLE,
    degenerate      BOOLEAN,
    -- One column per registered criterion, written from the criteria registry so
    -- adding a criterion cannot silently fail to be evaluated.
    fired_mhra_prr          BOOLEAN,
    fired_ema_ror025        BOOLEAN,
    fired_who_oe025         BOOLEAN,
    fired_dubious_prr_only  BOOLEAN,
    n_criteria_firing       INTEGER,
    PRIMARY KEY (run_id, drug_unii, reaction, as_of_quarter)
);

CREATE TABLE IF NOT EXISTS stat_pair_outcome (
    run_id          VARCHAR NOT NULL,
    drug_unii       VARCHAR NOT NULL,
    reaction        VARCHAR NOT NULL,
    criterion       VARCHAR NOT NULL,
    status          VARCHAR NOT NULL,
    signal_quarter  VARCHAR,
    label_quarter   VARCHAR,
    lead_time_quarters INTEGER,
    censored        BOOLEAN,
    in_survival_set BOOLEAN,
    is_event        BOOLEAN,
    firing_fraction DOUBLE,
    PRIMARY KEY (run_id, drug_unii, reaction, criterion)
);

CREATE TABLE IF NOT EXISTS stat_leakage (
    run_id          VARCHAR NOT NULL,
    drug_unii       VARCHAR NOT NULL,
    reaction        VARCHAR NOT NULL,
    statistic       VARCHAR NOT NULL,
    at_signal_onset DOUBLE,
    at_label_change DOUBLE,
    retrospective   DOUBLE,
    inflation_ratio DOUBLE,
    reports_before_label BIGINT,
    reports_after_label  BIGINT,
    notoriety_surge_ratio DOUBLE,
    notoriety_window_quarters INTEGER,
    is_notorious    BOOLEAN,
    PRIMARY KEY (run_id, drug_unii, reaction, statistic)
);

CREATE TABLE IF NOT EXISTS stat_diagnostics (
    run_id          VARCHAR NOT NULL,
    drug_unii       VARCHAR NOT NULL,
    reaction        VARCHAR NOT NULL,
    reporter_concentration DOUBLE,
    top_country     VARCHAR,
    top_country_share DOUBLE,
    consumer_share  DOUBLE,
    lawyer_share    DOUBLE,
    spike_ratio     DOUBLE,
    spike_quarter   VARCHAR,
    quarters_with_reports INTEGER,
    single_quarter_dependence DOUBLE,
    robustness_score DOUBLE,
    artefact_flags  VARCHAR,
    PRIMARY KEY (run_id, drug_unii, reaction)
);

CREATE TABLE IF NOT EXISTS model_survival_curve (
    run_id          VARCHAR NOT NULL,
    criterion       VARCHAR NOT NULL,
    n_pairs         INTEGER,
    n_events        INTEGER,
    n_censored      INTEGER,
    median_quarters_to_label DOUBLE,
    labelled_by_1y  DOUBLE,
    labelled_by_2y  DOUBLE,
    labelled_by_3y  DOUBLE,
    PRIMARY KEY (run_id, criterion)
);

CREATE TABLE IF NOT EXISTS model_survival_points (
    run_id          VARCHAR NOT NULL,
    criterion       VARCHAR NOT NULL,
    quarters        DOUBLE NOT NULL,
    survival        DOUBLE NOT NULL,
    PRIMARY KEY (run_id, criterion, quarters)
);

CREATE TABLE IF NOT EXISTS model_hazard_report (
    run_id          VARCHAR NOT NULL,
    n_train_rows    INTEGER,
    n_test_rows     INTEGER,
    n_train_events  INTEGER,
    n_test_events   INTEGER,
    holdout_from_quarter VARCHAR,
    roc_auc         DOUBLE,
    average_precision DOUBLE,
    base_rate       DOUBLE,
    precision_at_10 DOUBLE,
    precision_at_50 DOUBLE,
    precision_at_100 DOUBLE,
    lift_at_50      DOUBLE,
    fitted          BOOLEAN,
    note            VARCHAR,
    PRIMARY KEY (run_id)
);

CREATE TABLE IF NOT EXISTS model_hazard_coefficient (
    run_id          VARCHAR NOT NULL,
    feature         VARCHAR NOT NULL,
    coefficient     DOUBLE NOT NULL,
    PRIMARY KEY (run_id, feature)
);

CREATE TABLE IF NOT EXISTS model_calibration (
    run_id          VARCHAR NOT NULL,
    decile          INTEGER NOT NULL,
    mean_predicted  DOUBLE,
    observed_rate   DOUBLE,
    n_rows          INTEGER,
    PRIMARY KEY (run_id, decile)
);

-- Current-quarter ranked queue: the operational output. One row per open label
-- gap, scored by the hazard model and qualified by the artefact diagnostics.
CREATE TABLE IF NOT EXISTS model_priority_queue (
    run_id          VARCHAR NOT NULL,
    drug_unii       VARCHAR NOT NULL,
    drug_name       VARCHAR NOT NULL,
    reaction        VARCHAR NOT NULL,
    as_of_quarter   VARCHAR NOT NULL,
    hazard_probability DOUBLE,
    a BIGINT,
    ebgm DOUBLE,
    eb05 DOUBLE,
    ror DOUBLE,
    ror_ci_lower DOUBLE,
    prr DOUBLE,
    signal_quarter  VARCHAR,
    quarters_since_signal INTEGER,
    criteria_firing VARCHAR,
    robustness_score DOUBLE,
    artefact_flags  VARCHAR,
    rank            INTEGER,
    PRIMARY KEY (run_id, drug_unii, reaction)
);
