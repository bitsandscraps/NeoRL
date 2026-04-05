import copy
import itertools
from neorl.neorl_envs.logistics_distribution.ld_env import LogisticsDistributionEnv


log_path = 'log/tsp-v0/dqn/'


def cal_optim(env: LogisticsDistributionEnv):
    """
    Theoretically optimal strategy
    :return: (optim_return, length)
    """
    points = list(env.points.keys())
    envs = [copy.deepcopy(env) for _ in range(6)]
    _res = []
    for i, traj in enumerate(itertools.permutations(points)):
        ret, len_traj, d = 0, 0, False
        for a, b in traj:
            if (envs[i].unfinished == 0) or d:
                break
            if envs[i].points[(a, b)]:
                s, r, d, info = envs[i].step(a * env.LENGTH + b)
                len_traj += 1
                ret += r
        _res.append((ret, len_traj))
    optim_return, optim_length = sorted(_res, key=lambda x: x[0], reverse=True)[0]
    return optim_return, optim_length


def print_res(level, res):
    print("-" * 5 + level + "-" * 5)
    for logged_return, test_mean, test_length_mean in sorted(res, key=lambda x: x[1], reverse=True):
        print(f"{logged_return}\t| {test_mean:.3f}\t {test_length_mean:.0f}")
    print('\n')
