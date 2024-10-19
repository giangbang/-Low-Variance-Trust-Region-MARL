

def compose_logging(**kwargs):
    all_loggings = [count_backtrack]
    for logging in all_loggings:
        logging(**kwargs)


def count_backtrack(actors, logger, n_step, **kwargs):
    if getattr(actors[0], "n_backtrack", None) is None:
        return
    num_agents = len(actors)
    cnt_backtracks = np.empty(num_agents, dtype=np.float32)
    for actor in actors:
        cnt_backtracks[actor.i] = actor.n_backtrack

    n_backtracks_dict = dict(zip(
        [f"update_{i}th" for i in range(len(self.actor))],
        cnt_backtracks
    ))
    logger.writter.add_scalars("n_backtrack", n_backtracks_dict, n_step)
