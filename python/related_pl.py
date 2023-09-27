from timing import lap, finish

lap()
import polars as pl


def main():
    lap()
    posts = pl.read_json("../posts.json")
    lap()

    posts = posts.lazy()
    by_tag = posts.with_row_count().explode("tags").select("tags", "row_nr")
    pairs = by_tag.join(by_tag, on="tags")
    pairs = pairs.filter(pl.col("row_nr") != pl.col("row_nr_right"))

    counts = pairs.group_by(["row_nr", "row_nr_right"]).count()
    top5 = counts.sort("count", descending=True).group_by("row_nr_right", maintain_order=True).head(5).drop("count")

    related = posts.with_row_count().join(top5, on="row_nr")
    related = related.group_by("row_nr_right").agg(related=pl.struct("_id", "title", "tags"))
    related = related.rename({"row_nr_right": "row_nr"})

    result = posts.with_row_count().join(related, on="row_nr").select("_id", "tags", "related")
    result = result.collect()

    lap()
    result.write_json("../related_posts_python_pl.json", row_oriented=True)
    lap()
    finish()


if __name__ == "__main__":
    main()
