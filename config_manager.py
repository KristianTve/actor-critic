import configparser


class config_manager:
    def __init__(self):
        self.config = configparser.ConfigParser()
        #self.config.read('config_HAN.ini')
        self.config.read('config_HAN_NN.ini')

        #self.config.read('config_CAR.ini')
        #self.config.read('config_CAR_NN.ini')

        #self.config.read('config_GAM.ini')
        #self.config.read('config_GAM_NN.ini')

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
        episodes = self.config['GLOBALS']['episodes']
        time_steps = self.config['GLOBALS']['time_steps']

        return float(critic_lr), float(actor_lr), float(discount), float(trace_decay), float(epsilon), int(episodes), int(time_steps)

    def get_main_params(self):
        mode = self.config['MAIN']['mode']
        nn = self.config.getboolean('MAIN', 'NN')

        return mode, nn

    def get_hanoi_params(self):
        pegs = self.config['HANOI']['pegs']
        discs = self.config['HANOI']['discs']

        return int(pegs), int(discs)

    def get_cartpole_params(self):
        pole_length = self.config['CARTPOLE']['pole_length']
        pole_mass = self.config['CARTPOLE']['pole_mass']
        gravity = self.config['CARTPOLE']['gravity']
        timestep = self.config['CARTPOLE']['timestep']

        return float(pole_length), float(pole_mass), float(gravity), float(timestep)

    def get_gambler_params(self):
        win_prob = self.config['GAMBLER']['win_prob']

        return float(win_prob)

