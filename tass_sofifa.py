import pandas as pd

class TassSofia:
    def __init__(self, path2dataset):
        self.dataset = pd.read_csv(path2dataset, index_col = None)
        
        self.footballer_info = self.dataset[['Name', 'Country', 'Club', 'Id','Salary', 'Value']]
        self.footballer_atr = self.dataset.loc[:,~self.dataset.columns.isin(
                                            ['Country', 'Club', 'Id','Salary', 'Value'])]
        
    def get_footballer_atr(self, name, col = 'Name'):
        row = self.footballer_atr[self.footballer_atr[col] == name]
        return row.loc[:,~row.columns.isin(['Name'])]
    
    def get_footballer_info(self, name, col = 'Name'):
        return self.footballer_info[self.footballer_info[col] == name]
    

def main():
    names = ['H. Kane', 'T. Alexander-Arnold', 'R. Sterling', 'K. Walker',
             'P. Foden', 'R. James', 'J. Grealish', 'K. Trippier',
             'D. Rice', 'J. Maddison', 'B. Saka', 'M. Mount']
    
    so = TassSofia('sofifa_processed.csv')
    
    footballers = {}
    
    for name in names:
      footballers[name] = so.get_footballer_atr(name).values.flatten()
      
# if __name__ == '__main__':
#     main()
