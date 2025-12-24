CREATE DATABASE aidev CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE aidev;
SELECT USER(), CURRENT_USER();
USE aidev;
SHOW TABLES;
SELECT COUNT(*) FROM pull_request;
DESCRIBE pull_request;
DESCRIBE pr_commits;
DESCRIBE pr_comments;
DESCRIBE pr_reviews;
DESCRIBE pr_review_comments_v2;
DESCRIBE pr_commits;

SHOW COLUMNS FROM pull_request;
SHOW COLUMNS FROM pr_comments;
SHOW COLUMNS FROM pr_reviews;
SHOW COLUMNS FROM pr_review_comments_v2;
SHOW COLUMNS FROM pr_commits;
SHOW COLUMNS FROM aidev_user;

SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'aidev'
  AND table_name IN (
    'pull_request','pr_comments','pr_reviews','pr_review_comments_v2',
    'pr_commits','pr_timeline','aidev_user'
  )
ORDER BY table_name, ordinal_position;

USE aidev;
SHOW TABLES;
SELECT COUNT(*) FROM pr_review_comments_v2;



USE aidev;
-- 建索引
CREATE INDEX idx_pull_request_id ON pull_request(id);

CREATE INDEX idx_pr_comments_prid ON pr_comments(pr_id);
CREATE INDEX idx_pr_reviews_prid ON pr_reviews(pr_id);
CREATE INDEX idx_pr_reviews_id ON pr_reviews(id);

CREATE INDEX idx_pr_rc_reviewid ON pr_review_comments_v2(pull_request_review_id);

CREATE INDEX idx_pr_commits_prid ON pr_commits(pr_id);

# if PR is by human
DROP TABLE IF EXISTS rq1_user_type_map;
CREATE TABLE rq1_user_type_map AS
SELECT
  user,
  CASE
    WHEN SUM(user_type = 'User') > 0 THEN 'User'
    WHEN SUM(user_type = 'Bot')  > 0 THEN 'Bot'
    ELSE NULL
  END AS user_type
FROM (
  SELECT user, user_type
  FROM pr_comments
  WHERE user IS NOT NULL AND user_type IS NOT NULL

  UNION ALL

  SELECT user, user_type
  FROM pr_reviews
  WHERE user IS NOT NULL AND user_type IS NOT NULL

  UNION ALL

  SELECT user, user_type
  FROM pr_review_comments_v2
  WHERE user IS NOT NULL AND user_type IS NOT NULL
) t
GROUP BY user;

CREATE INDEX idx_rq1_user_type_user ON rq1_user_type_map(`user`(191));

-- has human comment
DROP TABLE IF EXISTS rq1_human_comment;
CREATE TABLE rq1_human_comment AS
SELECT pr_id, 1 AS has_human_comment
FROM (
  -- 普通评论
  SELECT pr_id
  FROM pr_comments
  WHERE user_type = 'User'

  UNION

  -- 行内评论：review_comment -> pr_reviews -> pr_id
  SELECT rv.pr_id
  FROM pr_review_comments_v2 rc
  JOIN pr_reviews rv
    ON rc.pull_request_review_id = rv.id
  WHERE rc.user_type = 'User'
) x
GROUP BY pr_id;

-- has human review
DROP TABLE IF EXISTS rq1_human_review;
CREATE TABLE rq1_human_review AS
SELECT pr_id, 1 AS has_human_review
FROM pr_reviews
WHERE user_type = 'User'
GROUP BY pr_id;

-- has human commit
DROP TABLE IF EXISTS rq1_human_commit;
CREATE TABLE rq1_human_commit AS
SELECT c.pr_id, 1 AS has_human_commit
FROM pr_commits c
LEFT JOIN rq1_user_type_map ua ON c.author = ua.user
LEFT JOIN rq1_user_type_map uc ON c.committer = uc.user
WHERE (ua.user_type = 'User' OR uc.user_type = 'User')
GROUP BY c.pr_id;

-- pr_scenarios_rq1
DROP TABLE IF EXISTS pr_scenarios_rq1;
CREATE TABLE pr_scenarios_rq1 AS
SELECT
  pr.id AS pr_id,
  pr.repo_id,
  pr.number,
  pr.agent,
  pr.state,
  pr.created_at,
  pr.merged_at,
  pr.closed_at,
  COALESCE(hc.has_human_comment, 0) AS has_human_comment,
  COALESCE(hr.has_human_review, 0)  AS has_human_review,
  COALESCE(hm.has_human_commit, 0)  AS has_human_commit,
  CASE
    WHEN COALESCE(hm.has_human_commit, 0) = 1 THEN 'S2_Human_coedited'
    WHEN (COALESCE(hc.has_human_comment, 0) = 1 OR COALESCE(hr.has_human_review, 0) = 1) THEN 'S1_Human_reviewed'
    ELSE 'S0_Solo_agent'
  END AS scenario_label
FROM pull_request pr
LEFT JOIN rq1_human_comment hc ON pr.id = hc.pr_id
LEFT JOIN rq1_human_review  hr ON pr.id = hr.pr_id
LEFT JOIN rq1_human_commit  hm ON pr.id = hm.pr_id;

-- sanity check
SELECT scenario_label, COUNT(*) AS n
FROM pr_scenarios_rq1
GROUP BY scenario_label
ORDER BY n DESC;

SELECT agent, scenario_label, COUNT(*) AS n
FROM pr_scenarios_rq1
GROUP BY agent, scenario_label
ORDER BY agent, n DESC;

SELECT
  AVG(has_human_comment) AS p_human_comment,
  AVG(has_human_review)  AS p_human_review,
  AVG(has_human_commit)  AS p_human_commit
FROM pr_scenarios_rq1;









