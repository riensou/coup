from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

        self.log_action_threshold = 100
        self._actions = {f"{action_type}_actions": 0 for action_type in ['income', 'foreign_aid', 'tax', 'steal', 'coup', 'assassinate', 'exchange']}

        self.log_block_threshold = 100
        self._blocks = {'accept_blocks': 0, 'dispute_blocks': 0, 'block_blocks': 0}

        self.log_dispose_threshold = 100
        self._disposes = {f"{role}_disposes": 0 for role in ['duke', 'assassin', 'captain', 'ambassador', 'contessa']}

        self.log_keep_threshold = 100
        self._keeps = {f"{role}_keeps": 0 for role in ['duke', 'assassin', 'captain', 'ambassador', 'contessa']}

        self.log_bluff_threshold = 100
        self._bluffs = {f"{move}_bluffs": 0 for move in ['total', 'tax', 'steal', 'assassinate', 'exchange', 'block']}
        self._bluffs |= {f"{move}_truths": 0 for move in ['total', 'tax', 'steal', 'assassinate', 'exchange', 'block']}


    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        if 'is_success' in info.keys():
            for k in info.keys():
                if k in ['is_success', 'episode', 'TimeLimit.truncated', 'terminal_observation']:
                    continue
                else:
                    if "actions" in k:
                        self._actions[k] += info[k]
                    elif "blocks" in k:
                        self._blocks[k] += info[k]
                    elif "disposes" in k:
                        self._disposes[k] += info[k]
                    elif "keeps" in k:
                        self._keeps[k] += info[k]
                    elif "bluffs" in k or "truths" in k:
                        self._bluffs[k] += info[k]
        return True
    
    def _on_rollout_end(self) -> None:
        super()._on_rollout_end()

        total_num_actions = sum([self._actions[k] for k in self._actions.keys()])
        total_num_blocks = sum([self._blocks[k] for k in self._blocks.keys()])
        total_num_disposes = sum([self._disposes[k] for k in self._disposes.keys()])
        total_num_keeps = sum([self._keeps[k] for k in self._keeps.keys()])
        total_num_bluffs = sum([self._bluffs[k] for k in self._bluffs.keys()])

        if total_num_actions >= self.log_action_threshold:
            for k in self._actions.keys():
                self.logger.record("action_rates/"+k[:k.rindex('_')]+"_rate", self._actions[k] / total_num_actions)
            self._actions = {f"{action_type}_actions": 0 for action_type in ['income', 'foreign_aid', 'tax', 'steal', 'coup', 'assassinate', 'exchange']}

        if total_num_blocks >= self.log_block_threshold:
            for k in self._blocks.keys():
                self.logger.record("block_rates/"+k[:k.rindex('_')]+"_rate", self._blocks[k] / total_num_blocks)
            self._blocks = {'accept_blocks': 0, 'dispute_blocks': 0, 'block_blocks': 0}

        if total_num_disposes >= self.log_dispose_threshold:
            for k in self._disposes.keys():
                self.logger.record("dispose_rates/"+k[:k.rindex('_')]+"_rate", self._disposes[k] / total_num_disposes)
            self._disposes = {f"{role}_disposes": 0 for role in ['duke', 'assassin', 'captain', 'ambassador', 'contessa']}

        if total_num_keeps >= self.log_keep_threshold:
            for k in self._keeps.keys():
                self.logger.record("keep_rates/"+k[:k.rindex('_')]+"_rate", self._keeps[k] / total_num_keeps)
            self._keeps = {f"{role}_keeps": 0 for role in ['duke', 'assassin', 'captain', 'ambassador', 'contessa']}

        if total_num_bluffs >= self.log_bluff_threshold:
            for k in self._bluffs.keys():
                if "bluffs" in k:
                    num_moves = self._bluffs[k] + self._bluffs[k[:k.rindex('_')]+"_truths"]
                    if num_moves:
                        self.logger.record("bluff_rates/"+k[:k.rindex('_')]+"_rate", self._bluffs[k] / num_moves)
            self._bluffs = {f"{move}_bluffs": 0 for move in ['total', 'tax', 'steal', 'assassinate', 'exchange', 'block']}
            self._bluffs |= {f"{move}_truths": 0 for move in ['total', 'tax', 'steal', 'assassinate', 'exchange', 'block']}