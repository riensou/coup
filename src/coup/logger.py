from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._stats_window_size = 100
        self._i = 0

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        if 'is_success' in info.keys():
            for k in info.keys():
                if k in ['is_success', 'episode', 'TimeLimit.truncated', 'terminal_observation']:
                    continue
                else:
                    if "actions" in k:
                        self._actions[k] += info[k]
        return True
    
    def _on_rollout_start(self) -> None:
        super()._on_rollout_start()
        if not self._i:
            self._actions = {f"{action_type}_actions": 0 for action_type in ['income', 'foreign_aid', 'tax', 'steal', 'coup', 'assassinate', 'exchange']}

    def _on_rollout_end(self) -> None:
        super()._on_rollout_end()

        self._i += 1

        if self._i == self._stats_window_size:

            self._i = 0

            # log action rates
            total_num_actions = sum([self._actions[k] for k in self._actions.keys()])
            if total_num_actions:
                for k in self._actions.keys():
                    self.logger.record("action_rates/"+k[:k.index('_')]+"_rates", self._actions[k] / total_num_actions)