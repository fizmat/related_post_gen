from timing import lap, finish

lap()
import pandas as pd


def main():
    lap()
    posts = pd.read_json("../posts.json")
    lap()

    by_tag = posts.tags.map(list).explode().reset_index()
    pairs = pd.merge(by_tag, by_tag, on="tags")
    pairs = pairs.where(pairs.index_x != pairs.index_y)

    counts = pairs.groupby(["index_x", "index_y"], sort=False).count()
    top5 = counts.sort_values("tags").groupby(level=0, group_keys=False, sort=False).tail(5).drop(columns="tags")

    related = top5.reset_index(level=0).join(posts).set_index("index_x")
    related = related.groupby(level=0).apply(lambda g: g.to_dict("records"))

    result = posts[["_id", "tags"]]
    result["related"] = related

    lap()
    result.to_json("../related_posts_python_pd.json", "records")
    lap()
    finish()


if __name__ == "__main__":
    main()
