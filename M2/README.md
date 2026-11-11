



AI Disclosure, AI ways used to help create these store\_sale\_type\_lookup.py,  sale type day table.py, make\_all\_variables\_mi\_matrix\_updated.py, compare\_stores\_stockouts\_and\_profit.py





&#x20;install: python -m pip install pandas numpy scikit-learn matplotlib

the sale type day table.py 

this generates a table of 7300 data points for the inferred sale type for each day and store in a csv file







store\_sale\_type\_lookup.py 



you run python store\_sale\_type\_lookup.py 



to get the JSON file, each store has it's dates and inferred sale type as a value and the store number can be used as a key to access its 730 values.



to see the store and all values run python store\_sale\_type\_lookup.py --store 104 --all 







**Data used for make\_all\_variables\_mi\_matrix\_updated.py, compare\_stores\_stockouts\_and\_profit.py:**



date: calendar date for the store day



store\_id: the store identifier



sale\_week\_type: 0=low promotion, 1=ad week, 2= big sale



the nine \*\_profit columns: the daily profit for every department



the nine zero\_ratio\_\* columns: the fraction of inventory observations with zero opening stock for each department.





The different departments are: Bakery, Beverages, Dairy, Frozen, Grocery, Health \& Beauty, Household, Meat \& Seafood, and Produce.







**Shared calculations:**



For store s on day d, the scripts show:



formula for department profit{



total\_department\_profit(s,d)= sum of the 9 departments profits



}





Average department zero stock ratio{



mean\_department\_zero\_ratio(s,d)= sum of the 9 zero ratio values/9



}



Average department zero stock percentage{



average\_department\_zero\_stock\_pct(s,d)= mean\_department\_zero\_ratio(s,d)\* 100



}







**what does make\_all\_variables\_mi\_matrix\_updated.py do?** 



It makes a 25\*25 normalized mutual information matrix that covers:



9 department profit variables

9 department zero stock ratio variables

store ID and sale week type

year, month ,and weekday

total\_department\_profit and mean\_department\_zero\_ratio



Not every day is included since it could lead to confusion so 60% of the data was used to make the MI matrix. You can though do --all-data to include all data. (python  make\_all\_variables\_mi\_matrix\_updated.py --all-data) I would recommend adding --exclude-Christmas after data since the stores are closed that day if you choose to represent all of the data.





**How the MI was calculated:**



the variables store, promotion category, year, month, weekday are kept as a category. Numerical variables are discretized (makes and equal number of bins to compare where bins contain the groups of rows) before the MI is calculated. the default is 5 bins. bins for Zero stock measurements, exact zero is placed in a separate group, the positive values are divided up into four groups. The values may reduce the number of groups.



The output of the program is a 0-1 scale. the higher the value means there is more dependence between variables. if there is a 1 then it is being compared to itself.



Ex. a high MI between produce and dairy stockout ratios would mean that the department's daily shortage patterns share information. It does not prove that one department's shortage caused the other department to have a shortage. Vice versa for low MI.





**The MI CSV files outputted:**



mi\_all\_features\_matrix.csv: the 25\*25 matrix. the values recorded to 5 decimal places. the first column identifies each row feature.





mi\_all\_pairs\_ranked.csv: all 300 distinct feature pairs, from highest to lowest MI. This doesn't repeat symmetric pairs or self-comparisons.



mi\_hypothesis\_relationships.csv: Selected promotion vs. stockout, stockout vs. same department profit, promotion vs. profit, and store wide summary comparisons relevant our research questions.





comparison\_by\_sale\_type.csv: A descriptive non-MI table. Store count, mean total department profit, and mean of the department zero stock percentages for each promotion category. This does use the same date subset as the MI analysis.



features\_used.csv: Lists every feature, feature from the csv or derived, its number of discrete categories, and its binning method.

















**compare\_stores\_stockouts\_and\_profit.py what does this do?**



This program uses the daily numerical values and not the MI values, and it excludes Christmas.



the first comparison:



the program records store\_id and sale\_week\_type. each group calculates the number of days, mean and median of the average department stockout percentage, and mean and median of total daily department profit.







formulas:



mean stockout percentage for a store/ promotion category{



mean\_stockout\_pct(s,t)= sum of daily stockout percentages for store s/ number of days in sale type t



}



Mean daily profit for that store / category{



mean\_profit(s,t)= sum of daily department profit for store s/ number of days in sale type t



}



big sale minus low promotion stockout difference{



stockout\_difference(s)= mean\_stockout\_pct(s,big sale)- mean\_stockout\_pct(s,low promotion)



}



big sale minus low promotion profit difference{



profit\_difference(s)= mean\_profit(s, big sale) - mean\_profit(s, low promotion)



(if the result is negative that means the average daily profit was lower during big sale weeks for that store)



}





comparison\_1\_stockout\_heatmap.png:



the x-axis is low promotion, ad week, and big sale.



the y-axis is the 10 store ID's.



Each square is the average opening stockout percent for the store during the promotion category.



dark blue is large average stockout percentage.





the CSV files:



comparison\_1\_big\_sale\_minus\_low\_by\_store.csv: 10 rows one per store. the low promotion and big sale averages, days included, and their stockout/profit differences.



comparison\_1\_each\_store\_by\_sale\_type.csv: (10 stores \* 3 sales types) 30 groups. number of days, mean/median stockout %, and mean/median daily profit.





**The second comparison: high vs low stockout days within each store and sale type:**



The program separates programs into groups sharing the same store and same promotion category. It then computes the median daily stockout percentage separately withing each group.



formulas:



median\_cutoff(s,t)=median of the daily stockout percentages for store s and sale type t



Each day is assigned to{



low stockout group: daily stockout percentage <= group median

high stockout group: daily stockout percentage > group median



}





the profit difference for each group{



difference(s,t)= mean(profit on high stockout days)- mean(profit on low stockout days)



}





the equal weight average of eligible stores differences for each sale week type{



mean\_difference(t)=sum of difference(s,t) across eligible stores / number of eligible stores



}



if a store/category is omitted from this summary then it does not have any high stock out days. 







Graph: comparison\_2\_profit\_differences.png:



the x-axis is the sale week type.



the y-axis is the mean profit difference (money per store day)= high stockout days minus low stockout days.



the bar decreases with lower average profit on high stockout days.





csv files:



comparison\_2\_high\_vs\_low\_stockout\_by\_store\_sale.csv: 30 store/category groups with every groups median threshold, low/high day counts, low/high stockout means, low/high profit means, and profit difference.



comparison\_2\_summary\_by\_sale\_type.csv: 3 sale categories. equal weight averages across eligible store groups, these values make the profit difference bar chart.





daily\_store\_features\_for\_comparison.csv: one row per store day showing date, store, sale category, derived average stockout %, and summed department profit. this can be helpful for checking our calculations.









**conclusion: the big sale weeks showed higher average opening stockout percentages, and the high stockout days also had lower average department profit.**

















