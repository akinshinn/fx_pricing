from simulation import CIR
import yaml 

class Pricing:
    def __init__(self, lower_bound, upper_bound, nominal, start_date, end_date):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.nominal = nominal
        self.start_date = start_date
        self.end_date = end_date

        self.model = None

    def price(self, M, confidence=0.95, time_steps = 1):
        return self.model.get_monte_carlo_estimate(M, confidence)

    def get_model(self, config_path): 
        with open(config_path, 'r') as file: 
            config = yaml.safe_load(file)

        self.model = CIR(
            kappa=config['simulation']['kappa'],
            theta=config['simulation']['theta'],
            sigma=config['simulation']['sigma'],
            r0=config['simulation']['r0'],
            T=config['simulation']['T'],
            N=config['simulation']['N']
        )
        