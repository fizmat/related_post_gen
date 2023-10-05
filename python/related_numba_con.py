from timing import lap, finish

lap()
import numpy as np
import orjson
from numba import njit, prange


@njit(parallel=True)
def get_all_top5(n_posts, t_to_pp, t_to_np, p_to_tt, p_to_nt):
    all_top5 = np.empty((n_posts, 5), np.uint16)
    for current_p in prange(n_posts):
        tt = p_to_tt[current_p]
        relation_count = np.zeros(n_posts, dtype=np.uint8)
        for tj in range(p_to_nt[current_p]):
            for pi in range(t_to_np[tt[tj]]):
                relation_count[t_to_pp[tt[tj], pi]] += 1
        relation_count[current_p] = 0

        # get top 5:
        # (probably can be optimized more)
        max_i = np.zeros(5, dtype=np.uint16)
        max_r = np.zeros(5, dtype=np.uint8)
        for i, r in enumerate(relation_count):
            for j in range(4, -1, -1):
                if r > max_r[j]:
                    max_r[:j] = max_r[1:j + 1]
                    max_i[:j] = max_i[1:j + 1]
                    max_r[j] = r
                    max_i[j] = i
                    break
        all_top5[current_p] = max_i
    return all_top5


def precompile():
    lap()
    # JIT compile by running with the arguments of the correct type
    # 1) measure compile time
    # 2) get correct processing (without compilation) time using the cached machine code
    get_all_top5(
        0,
        np.empty((0, 0), dtype=np.uint16),
        np.empty(0, dtype=np.uint16),
        np.empty((0, 0), dtype=np.uint8),
        np.empty(0, dtype=np.uint8),
    )


def main():
    lap()
    with open("../posts.json", "rb") as f:
        s = f.read()
        posts = orjson.loads(s)
    lap()

    unique_tags = set(tag for post in posts for tag in post["tags"])
    tag_to_t = {t: np.uint8(i) for i, t in enumerate(unique_tags)}
    t_to_pp_list = [[] for _ in unique_tags]
    p_to_nt = np.array([len(post["tags"]) for post in posts], dtype=np.uint8)
    p_to_tt = np.empty((len(posts), p_to_nt.max()), dtype=np.uint8)

    for p, post in enumerate(posts):
        for tj, tag in enumerate(post["tags"]):
            t_to_pp_list[tag_to_t[tag]].append(p)
            p_to_tt[p, tj] = tag_to_t[tag]
    t_to_np = np.empty(len(unique_tags), dtype=np.uint16)
    t_to_pp = np.empty((len(unique_tags), max(len(posts) for posts in t_to_pp_list)), dtype=np.uint16)
    for t, pp in enumerate(t_to_pp_list):
        t_to_np[t] = len(pp)
        t_to_pp[t, :len(pp)] = pp

    all_top5 = get_all_top5(len(posts), t_to_pp, t_to_np, p_to_tt, p_to_nt)

    all_related_posts = []
    for post, top5 in zip(posts, all_top5):
        all_related_posts.append(
            {
                "_id": post["_id"],
                "tags": post["tags"],
                "related": [posts[i] for i in top5],
            }
        )

    lap()
    with open("../related_posts_python_numba_con.json", "wb") as f:
        s = orjson.dumps(all_related_posts)
        f.write(s)
    lap()
    finish()


if __name__ == "__main__":
    precompile()
    main()
