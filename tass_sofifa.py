import pandas as pd

from utils import get_cos_similarity

class TassSofia:
    def __init__(self, path2dataset):
        self.dataset = pd.read_csv(path2dataset, index_col = None)
        
        self.footballer_info = self.dataset[['Name', 'Country', 'Club', 'Id']]
        self.footballer_atr = self.dataset.loc[:,~self.dataset.columns.isin(
                                            ['Country', 'Club', 'Id'])]
        
    def get_footballer_atr(self, name, col = 'Name'):
        row = self.footballer_atr[self.footballer_atr[col] == name]
        return row.loc[:,~row.columns.isin(['Name'])]
    
    def get_footballer_info(self, name, col = 'Name'):
        return self.footballer_info[self.footballer_info[col] == name]
    

def main():
    so = TassSofia('sofifa_processed.csv')
    
    names = ['R. Lewandowski', 'K. Benzema']
    footballers = {}
    
    for name in names:
      footballers[name] = so.get_footballer_atr(name).values  

    print(get_cos_similarity(footballers['R. Lewandowski'], footballers['K. Benzema']))


if __name__ == "__main__":
    main()