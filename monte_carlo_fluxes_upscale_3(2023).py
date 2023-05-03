# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:49:14 2023

@author: mmo990
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# First IQR to remove outliers
# Then do calcs that are in 
# Then k depending on if lake, pond, fluvial
# bootstrap dissolved gas concentrations
# calculate flux
# read-in area and confusion matrices to get area uncertainties
# calculcate emissions

class Fluxes:
    def __init__(self, gas, gas_air, wb_type, df=0, df_k600=0, k=0, bs_replicates=0, df_boot=0):
        self.gas = gas
        self.gas_air = gas_air
        self.wb_type = wb_type
        
    def subset_df(self, my_df):
        self.df = my_df[my_df['waterbody'].str.contains(self.wb_type)]
        self.df = self.df[[self.gas, self.gas_air, 'Date', 'Area', 'T', 'T - K', 'pH', 'EC', 'Salinity', 'year']]
        self.df['Date'] = pd.to_datetime(self.df['Date']).dt.date
        
    def outlier_removal(self, ll, ul, fpath_outliers):
        """
        outlier_removal removes outliers based on the 1.5IQR rule. 

        :param ll: lowerl limit of IQR
        :param ul: upper limit of IQR   
        :param fpath_outliers: filepath and name where csv file of removed outliers is saved
        :return: dataframe without outliers based on 1.5IQR
        """ 
    
        Q1 = self.df[self.gas].quantile(ll)
        Q3 = self.df[self.gas].quantile(ul)
        IQR = Q3 - Q1
        min_limit = (Q1 - 1.5*IQR)
        max_limit = (Q3 + 1.5*IQR)
        df_out_removed = self.df.loc[(self.df[self.gas] > min_limit) & (self.df[self.gas] < max_limit)]
        df_outliers = self.df[~self.df.astype(str).apply(tuple, 1).isin(df_out_removed.astype(str).apply(tuple, 1))]
        df_outliers.to_csv(fpath_outliers)
        self.df = df_out_removed

    def CO2_conc(self):
        # in g/mol
        atomic_weight_co2 = 44.01
        # from g/cm3 to g/L
        density_freshwater = 1 * 0.001
        
        # Convert from ppm in solution to micromol/L
        self.df['diss co2'] = self.df['diss CO2.ppm'] / atomic_weight_co2 * density_freshwater * 1000000
        
        return self.df
    
    def CO2_eq(self):
        """
        Calculating equivalent CO2 dissolved concentration of atmospheric values using Weiss, 1974 formula 

        """         
        self.df['CO2 eq'] = self.df['CO2 air']*(np.exp(-58.0931+90.5069*(100/self.df['T - K'])+22.294*np.log(self.df['T - K']/100)+self.df['Salinity']*(0.027766-0.025888*(self.df['T - K']/100)+0.0050578*np.square(self.df['T - K']/100))))
        return self.df
        
    def CH4_eq(self):
        """
        Calculating equivalent CH4 dissolved concentration of atmospheric values using Wiesenburg & Guinasso Jr, 1979 

        """         
        A1 = -415.2807
        A2 = 596.8104
        A3 = 379.2599
        A4 = -62.0757
        B1 = -0.059160
        B2 = 0.032174
        B3 = -0.0048198
        self.df['CH4 eq'] = np.exp(np.log(self.df['CH4 air']*0.000001) + A1 + A2*(100/self.df['T - K']) + A3*np.log(self.df['T - K']/100) + A4*(self.df['T - K']/100) + self.df['Salinity']*(B1 + B2*(self.df['T - K']/100) + B3*(np.square(self.df['T - K']/100))))*0.001

    def N2O_eq(self):
        """
        Calculating equivalent N2O dissolved concentration of atmospheric values using Weiss & Price, 1980 

        """         
        A1 = -165.8806
        A2 = 222.8743
        A3 = 92.0792
        A4 = -1.48425
        B1 = -0.056235
        B2 = 0.031619
        B3 = 0.0048472
        self.df['N2O eq'] = self.df['N2O air']*np.exp(A1 + A2*(100/self.df['T - K']) + A3*np.log(self.df['T - K']/100) + A4*np.square(self.df['T - K']/100) + self.df['Salinity']*(B1 + B2*(self.df['T - K']/100) + B3*(np.square(self.df['T - K']/100))))
                             
    def Sc(self):
        """
        Calculating Schmidt Number (Sc) based on polynomial from Wanninkhof, 2014
        
        :param gas: string to determine which polynomial to use
        """ 
        
        if self.gas == 'CO2':
            self.df['Sc freshwater'] = 1923.6 - 125.06 * self.df['T'] + 4.3773*np.power(self.df['T'], 2) - 0.085681*np.power(self.df['T'], 3) + 0.00070284*np.power(self.df['T'], 4)
            self.df['Sc saltwater'] = 2116.8 - 136.25 * self.df['T'] + 4.7353 * np.power(self.df['T'], 2) - 0.092307 * np.power(self.df['T'], 3) + 0.0007555 * np.power(self.df['T'], 4)
        elif self.gas == 'CH4':
            self.df['Sc freshwater'] = 1909.4 - 120.78 * self.df['T'] + 4.1555 * np.power(self.df['T'], 2) - 0.080578 * np.power(self.df['T'], 3) + 0.00065777 * np.power(self.df['T'], 4)
            self.df['Sc saltwater'] = 2101.2 - 131.54 * self.df['T'] + 4.4931 * np.power(self.df['T'], 2) - 0.08676 * np.power(self.df['T'], 3) + 0.00070663 * np.power(self.df['T'], 4)
        else:
            self.df['Sc freshwater'] = 2141.2 - 152.56 * self.df['T'] + 5.8963 * np.power(self.df['T'], 2) - 0.12411 * np.power(self.df['T'], 3) + 0.0010655 * np.power(self.df['T'], 4)
            self.df['Sc saltwater'] = 2356.2 - 166.38 * self.df['T'] + 6.3952 * np.power(self.df['T'], 2) - 0.13422 * np.power(self.df['T'], 3) + 0.0011506 * np.power(self.df['T'], 4)
        
        self.df['Sc'] = self.df['Sc freshwater'] + ((self.df['Sc saltwater'] - self.df['Sc freshwater']) / 35) * self.df['Salinity']
        
        return self.df 
    
    
    def k_cc(self, wind):
        """
        Calculates k [m/d] value using k600 [cm/h] based on relationship with wind from Cole & Caraco, 1998
        Only used for lake and flood data
        
        :param wind: dataframe with wind data

        """          
        # Calculate mean value for each date
        wind_means = wind.groupby('date').mean()
        
        # Convert each of the velocities to equivalent at 10m
        wind_10 = pd.DataFrame()
        
        # Height unit is meters
        H1 = 6
        H2 = 3.2
        H3 = 2
        
        a = 1/7
        
        wind_10['W1'] = wind_means['Wind Vel 1'] * math.pow((10/H1), a)
        wind_10['W2'] = wind_means['Wind Vel 2'] * math.pow((10/H2), a)
        wind_10['W3'] = wind_means['Wind Vel 3'] * math.pow((10/H3), a)
        
        wind_10_mean = wind_10.mean(axis=1)
        
        # Add the new column wind to the dataframe by matching the dates
        self.df = self.df.merge(wind_10_mean.rename('wind'), how='inner', left_on='Date', right_index=True)

        # Calculate k600        
        self.df['k600'] = 2.07 + 0.215 * np.power(self.df['wind'], 1.7) 
        
                
        # Calculate k
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            if row['wind']<3:
                self.df['k'] = (self.df['k600']*np.power((self.df['Sc']/600), 0.67)) / (100/24)
            else:
                self.df['k'] = (self.df['k600']*np.power((self.df['Sc']/600), 0.5)) / (100/24)
         
        k = self.df        
    
        return self.df, k


    def k_lit(self, fname):
        """
        Calculates k [m/d] value using k600 [cm/h] from literature
        Used for pond and fluvial fluxes

        """           
        self.df_k600 = pd.read_excel(fname, index_col=0, engine='openpyxl', skiprows=1)
        self.df_k600.reset_index(inplace=True)
        
        # Calculate k
        self.df_k600['k'] = (self.df_k600['k600']*np.power((self.df['Sc'].mean()/600), 0.5)) / (100/24)
        
        return self.df_k600     
   
    
    def bootstrap_replicate_1d(self, data, func):
        """Generate bootstrap replicate of 1D data."""
        
        bs_sample = np.random.choice(data, len(data))
        
        return func(bs_sample)
    
    
    def draw_bs_reps(self, data, func, size):
        """
        Draw bootstrap replicates
        """ 
        
        # Initialise array or replicates
        bs_replicates = np.empty(size)
        
        # Generate replicates
        for i in range(size):
            bs_replicates[i] = self.bootstrap_replicate_1d(data, func)
            
        return bs_replicates
  
    
    def mc_uniform(self, dis_n):
        """
        Monte Carlo uncertainties with uniform distribution for k values from literature
        to compute confidence intervals
        """ 
        
        np.random.seed(42)
        df_k = self.df_k600[self.df_k600['waterbody'].str.contains(self.wb_type)]
        k_o = df_k['k']
        
        # Draw out a uniform distribution with k values
        self.k = np.random.uniform(low=k_o.min(), high=k_o.max(), size=dis_n)
        
        # Plot the PDF and label axes
        plt.hist(self.k, bins=50, density=True, histtype='step')
        plt.xlabel(self.wb_type + ' k values [m/d]')
        plt.ylabel('CDF')
        
        # Show plot
        plt.show()
        
        # Create an ECDF from real data: x, y
        data = k_o.to_numpy()
        n = len(data)
        
        x = np.sort(data)
        y = np.arange(1, n+1) / n
        
        
        # Create a CDF from theoretical samples: x_theor, y_theor
        n_theor = len(self.k)
        
        x_theor = np.sort(self.k)
        y_theor = np.arange(1, n_theor+1) / n_theor
        
        
        # Overlay the plots
        plt.plot(x_theor, y_theor)
        plt.plot(x, y, marker='.', linestyle='none')
        
        plt.xlabel(self.wb_type + ' k values [m/d]')
        plt.ylabel('CDF')
        
        plt.show()        
        
        return self.k
    
    def bootstrapping(self, varble, boot_n):        
        gas_conc = self.df[varble].to_numpy()
        
        bs_replicates = self.draw_bs_reps(gas_conc, np.median, boot_n)
        
        # Make a histogram of the results
        plt.hist(bs_replicates, bins=50, density=True)
        plt.xlabel('median ' + varble)
        plt.ylabel('PDF')
        
        plt.show()
        
        # Compute the 95% confidence interval: conf_int
        conf_int = np.percentile(bs_replicates, [2.5, 97.5])
        print('95% confidence interval of =', varble, conf_int) 
        
        return bs_replicates
    
    def fluxes_calc(self, cs, ceq, ks, cols, flux_name):
        if self.wb_type == 'Lake':
            boot_data = {cols[0]: cs, cols[1]: ceq, cols[2]: ks}
            self.df_boot = pd.DataFrame(boot_data)
            self.df_boot[flux_name] = (self.df_boot[cols[0]] - self.df_boot[cols[1]]) * self.df_boot[cols[2]]
        else:
            boot_data = [cs, ceq, self.k]
            self.df_boot = pd.DataFrame(boot_data, columns=cols)
            self.df_boot[flux_name] = (self.df_boot[cols[0]] - self.df_boot[cols[1]]) * self.df_boot[cols[2]] 
        
        # Compute the 95% confidence interval: conf_int
        conf_int = self.df_boot[flux_name].quantile([0.025, 0.975])
        print('95% confidence interval of', flux_name, conf_int) 
                
        return self.df_boot


      

        
        
#class Emissions        

    

lake_co2 = Fluxes('diss CO2.ppm', 'CO2 air', 'Lake')
print(lake_co2.gas)
print(lake_co2.wb_type)
lake_co2.subset_df(df)
lake_co2.CO2_conc()
print(lake_co2.df)
lake_co2.outlier_removal(0.25, 0.75, 'C:/Users/mmo990/surfdrive/Paper1/EDA 2022/Outliers removed from up-scaling/pond_co2.csv')
lake_co2.CO2_eq()
lake_co2.Sc()
lake_co2.k_cc(df_wind)

lake_co2_s_boot = lake_co2.bootstrapping('diss co2', 1000)
lake_co2_eq_boot = lake_co2.bootstrapping('CO2 eq', 1000)
lake_co2_k_boot = lake_co2.bootstrapping('k', 1000)

lake_co2.fluxes_calc(lake_co2_s_boot, lake_co2_eq_boot, lake_co2_k_boot, ['diss co2', 'co2 eq', 'k'], 'flux co2')



pond_co2 = Fluxes('diss CO2.ppm', 'CO2 air', 'Pond')
print(pond_co2.gas)
print(pond_co2.wb_type)
pond_co2.subset_df(df)
print(pond_co2.df)
pond_co2.outlier_removal(0.25, 0.75, 'C:/Users/mmo990/surfdrive/Paper1/EDA 2022/Outliers removed from up-scaling/pond_co2.csv')
pond_co2.CO2_eq()
pond_co2.Sc()
pond_co2.k_lit('C:/Users/mmo990/surfdrive/Paper1/Data/k600_literature.xlsx')
pond_co2.mc_uniform(1000)
a = pond_co2.bootstrapping('diss CO2.ppm', 1000)

Pond_ch4 = Fluxes('diss CH4', 'CH4 air', 'Pond')
print(Pond_ch4.gas)
print(Pond_ch4.wb_type)
Pond_ch4.subset_df(df)
print(Pond_ch4.df)
Pond_ch4.outlier_removal(0.25, 0.75, 'C:/Users/mmo990/surfdrive/Paper1/EDA 2022/Outliers removed from up-scaling/pond_ch4.csv')
Pond_ch4.CH4_eq()


Pond_n2o = Fluxes('diss N2O', 'N2O air', 'Pond')
print(Pond_n2o.gas)
print(Pond_n2o.wb_type)
Pond_n2o.subset_df(df)
print(Pond_n2o.df)
Pond_n2o.outlier_removal(0.25, 0.75, 'C:/Users/mmo990/surfdrive/Paper1/EDA 2022/Outliers removed from up-scaling/pond_n2o.csv')
Pond_n2o.N2O_eq()



fluvial_co2 = Fluxes('diss CO2.ppm', 'CO2 air', 'Fluvial')
print(fluvial_co2.gas)
print(fluvial_co2.wb_type)
fluvial_co2.subset_df(df)
print(fluvial_co2.df)
fluvial_co2.outlier_removal(0.25, 0.75, 'C:/Users/mmo990/surfdrive/Paper1/EDA 2022/Outliers removed from up-scaling/fluvial_co2.csv')
fluvial_co2.CO2_eq()
fluvial_co2.Sc()
fluvial_co2.k_lit('C:/Users/mmo990/surfdrive/Paper1/Data/k600_literature.xlsx')
fluvial_co2.mc_uniform(1000)        
 
    
 
  # 3  def ecdf(self, df, col):
    #    """
   #      Compute ECDF for a one-dimensioanl array of measurements
    #     """     
     #    #df_sub = df[df['waterbody'].str.contains(self.wb_type)]
    #     data = df[col].to_numpy()
        
    #     n = len(data)
   #      
    #     x = np.sort(data)
    #     y = np.arange(1, n+1) / n
        
   #      plt.plot(x, y, marker='.', linestyle='none')
        
    #     plt.xlabel(col)
    #     plt.ylabel('ECDF')
        
    #     plt.show()
        
    #     return x, y