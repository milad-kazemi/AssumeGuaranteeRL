from workers.worker import Worker


class RuleBasedWorker(Worker):
    def __init__(self, constants, device, env, rule_set, data_collector, id):
        super(RuleBasedWorker, self).__init__(constants, device, env, id, data_collector)
        self.rule_set = rule_set

    def _reset(self):
        self.rule_set.reset()

    def _get_prediction(self, states, actions=None, ep_step=None):
        assert actions == None  # Doesnt want actions
        assert isinstance(states, dict)
        return {'a': self.rule_set(states)}

    def _get_action(self, prediction):
        return prediction['a']

    def _copy_shared_model_to_local(self):
        pass
