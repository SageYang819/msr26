USE aidev;

-- PR-level wide table
DROP TABLE IF EXISTS pr_rq12;
CREATE TABLE pr_rq12 AS
SELECT
  s.pr_id,
  s.repo_id,
  s.number,
  s.agent,
  s.state,
  s.created_at,
  s.merged_at,
  s.closed_at,
  s.has_human_comment,
  s.has_human_review,
  s.has_human_commit,
  s.scenario_label,
  c.review_count,
  c.request_changes_count,
  c.comment_count,
  c.post_review_review_count
FROM pr_scenarios_rq1 s
JOIN pr_cost_rq2 c
  ON s.pr_id = c.pr_id;

CREATE INDEX idx_pr_rq12_prid ON pr_rq12(pr_id);

-- Check
SELECT COUNT(*) AS n FROM pr_rq12;
