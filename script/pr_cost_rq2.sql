USE aidev;

-- 0) 快速确认当前库
SELECT DATABASE() AS current_db;

-- 1) review_count / request_changes_count / first_review_time
DROP TABLE IF EXISTS rq2_reviews;
CREATE TABLE rq2_reviews AS
SELECT
  pr_id,
  COUNT(*) AS review_count,
  SUM(state = 'CHANGES_REQUESTED') AS request_changes_count,
  MIN(submitted_at) AS first_review_time
FROM pr_reviews
GROUP BY pr_id;

CREATE INDEX idx_rq2_reviews_prid ON rq2_reviews(pr_id);

-- 2) comment_count（pr_comments + pr_review_comments_v2）
DROP TABLE IF EXISTS rq2_comments;
CREATE TABLE rq2_comments AS
SELECT
  pr_id,
  SUM(n) AS comment_count
FROM (
  SELECT pr_id, COUNT(*) AS n
  FROM pr_comments
  GROUP BY pr_id

  UNION ALL

  SELECT rv.pr_id AS pr_id, COUNT(*) AS n
  FROM pr_review_comments_v2 rc
  JOIN pr_reviews rv
    ON rc.pull_request_review_id = rv.id
  GROUP BY rv.pr_id
) t
GROUP BY pr_id;

CREATE INDEX idx_rq2_comments_prid ON rq2_comments(pr_id);

-- 3) 迭代成本 proxy：post-review review count = review_count - 1（同一 PR 的第2次及以后 review 数）
DROP TABLE IF EXISTS rq2_post_review_reviews;
CREATE TABLE rq2_post_review_reviews AS
SELECT
  pr_id,
  GREATEST(COUNT(*) - 1, 0) AS post_review_review_count
FROM pr_reviews
GROUP BY pr_id;

CREATE INDEX idx_rq2_postrev_prid ON rq2_post_review_reviews(pr_id);

-- 4) 汇总成本表 pr_cost_rq2
DROP TABLE IF EXISTS pr_cost_rq2;
CREATE TABLE pr_cost_rq2 AS
SELECT
  pr.id AS pr_id,
  COALESCE(rv.review_count, 0) AS review_count,
  COALESCE(rv.request_changes_count, 0) AS request_changes_count,
  COALESCE(cm.comment_count, 0) AS comment_count,
  COALESCE(prr.post_review_review_count, 0) AS post_review_review_count
FROM pull_request pr
LEFT JOIN rq2_reviews rv ON pr.id = rv.pr_id
LEFT JOIN rq2_comments cm ON pr.id = cm.pr_id
LEFT JOIN rq2_post_review_reviews prr ON pr.id = prr.pr_id;

CREATE INDEX idx_pr_cost_prid ON pr_cost_rq2(pr_id);

-- 5) 验证表确实生成了
SHOW TABLES LIKE 'pr_cost_rq2';

SELECT
  AVG(review_count) AS avg_reviews,
  AVG(request_changes_count) AS avg_changes_req,
  AVG(comment_count) AS avg_comments,
  AVG(post_review_review_count) AS avg_post_reviews
FROM pr_cost_rq2;


SHOW TABLES LIKE 'rq2_post_review_reviews';
SHOW TABLES LIKE 'pr_cost_rq2';


SELECT * FROM pr_cost_rq2 LIMIT 50;



