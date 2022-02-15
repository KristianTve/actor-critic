import configparser


class config_manager:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        # Initializing variables in config
        self.layers = None
        self.input_neurons = None
        self.activation_function = None
        self.learning_rate = None
        self.batch_size = None
        self.verbose = None

    def fetch_net_data(self):
        layer_size = []
        layer_act = []
        lr = []

        self.layers = self.config['GLOBALS']['Layers']
        self.input_neurons = self.config['GLOBALS']['Input_neurons']
        self.activation_function = self.config['GLOBALS']['Activation_function']
        self.batch_size = self.config['GLOBALS']['batch_size']
        self.verbose = self.config.getboolean('GLOBALS', 'verbose')
        optimizer = self.config['GLOBALS']['optimizer']

        for i in range(int(self.layers)):
            layer_size.append(int(self.config['LAYER'+str(i+1)]['neurons']))
            layer_act.append(self.config['LAYER'+str(i+1)]['act'])

        layer_act.append(self.config['OUTPUT']['act'])
        return (int(self.layers),
                int(self.input_neurons),
                int(self.batch_size),
                bool(self.verbose),
                layer_size,
                layer_act,
                optimizer)

    def fetch_actor_critic_data(self):
        critic_lr = self.config['GLOBALS']['critic_lr']
        actor_lr = self.config['GLOBALS']['actor_lr']
        discount = self.config['GLOBALS']['discount']
        trace_decay = self.config['GLOBALS']['trace_decay']
        epsilon = self.config['GLOBALS']['epsilon']

        return float(critic_lr), float(actor_lr), float(discount), float(trace_decay), float(epsilon)

    def get_main_params(self):
        mode = self.config['MAIN']['mode']
        nn = self.config.getboolean('MAIN', 'NN')

        return mode, nn
