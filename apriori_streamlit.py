import streamlit as st

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
from wordcloud import WordCloud,STOPWORDS
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
import networkx as nx 
from streamlit_bokeh_events import streamlit_bokeh_events
import bokeh.plotting as bpl
from bokeh.models import ColumnDataSource, CustomJS

#Apriori libraries 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def csv_data(csv_file_path):
    groceries=pd.read_csv(csv_file_path)
    groceries.rename(columns = {'Member_number':'id','itemDescription':'item'}, inplace = True)

    groceries['Date']= pd.to_datetime(groceries['Date'], format='%d-%m-%Y')
    groceries['year'] = groceries['Date'].apply(lambda x : x.year)
    groceries['month'] = groceries['Date'].apply(lambda x : x.month)
    groceries['day'] = groceries['Date'].apply(lambda x : x.day)
    groceries['weekday'] = groceries['Date'].apply(lambda x : x.weekday())
    groceries=groceries[['id', 'Date','year', 'month', 'day','weekday','item']]

    return groceries

def item_sold_each_month(groceries):
    # df1=groceries.groupby(['year']).filter(lambda x: (x['year'] == 2014).any())
    # df2=groceries.groupby(['year']).filter(lambda x: (x['year'] == 2015).any())

    # sales_2014=hv.Bars(df1.groupby(['month'])['item'].count()).opts(ylabel="# of items", title='# of items sold in 2014')
    # sales_2015=hv.Bars(df2.groupby(['month'])['item'].count()).opts(ylabel="# of items", title='# of items sold in 2015')

    # Filtering data by year 2014 and 2015
    df1 = groceries[groceries['year'] == 2014]
    df2 = groceries[groceries['year'] == 2015]

    # Plotting monthly data of the number of quantity purchased in 2014 and 2015
    chart1 = alt.Chart(df1.groupby(['month'])['item'].count().reset_index()).mark_bar().encode(
        x='month:N',
        y='item:Q',
    ).properties(
        width=380,
        height=300,
        title='# of items sold in 2014'
    )

    chart2 = alt.Chart(df2.groupby(['month'])['item'].count().reset_index()).mark_bar().encode(
        x='month:N',
        y='item:Q',
    ).properties(
        width=380,
        height=300,
        title='# of items sold in 2015'
    )

    # # Merging both plots
    # combined_chart = (chart1 + chart2).configure_axis(
    #     grid=True
    # ).configure_view(
    #     continuousWidth=500
    # )

    return chart1, chart2


    #Merging both plots
    # sales = (sales_2014 + sales_2015).opts(opts.Bars(width=380, height=300,tools=['hover'],show_grid=True))
    # hv.plot(sales, backend='bokeh').update(bokeh_plot)

    # source = ColumnDataSourece
    
    # sales_2014_bokeh = hv.render(sales_2014, backend='bokeh')
    # bokeh_plot = bpl.figure(width=400, height=300)
    # bokeh_plot = sales_2014_bokeh
    # source = ColumnDataSource(data={"x":[], "y":[]})
    # sales = bokeh_plot.circle("x", "y", source = source)
    return sales_2014

def altair(groceries):
    #Creating temporary data which has quantity purchased column
    temp=groceries.copy()
    temp['qty_purchased']=groceries['id'].map(groceries['id'].value_counts())

    #Slicing first 5000 rows as altair library can't plot any data which has record beyond that
    temp1=temp[:5000]

    #Plotting
    brush = alt.selection(type='interval', encodings=['x'])

    #Plotting the bar chart
    bars = alt.Chart().mark_bar(color="green").encode(
        x=alt.X('month(Date):O',title="Month"),
        y=alt.Y('mean(qty_purchased):Q',title="Last Price"),
        opacity=alt.condition(brush, alt.OpacityValue(1), alt.OpacityValue(0.7)),
        tooltip=['month(Date)','mean(qty_purchased)']
    ).add_selection(
        brush
    ).properties(height=400,width=600,title="Monthly quantity purchased from grocery store-Drag over bars and find average")

    #Plotting avrage line
    line = alt.Chart().mark_rule(color='firebrick').encode(
        y='mean(qty_purchased):Q',
        size=alt.SizeValue(3),
        tooltip=['mean(qty_purchased)']
    ).transform_filter(
        brush
    )

    #Display plot using sliced data
    result = alt.layer(bars, line, data=temp1)
    return result

def two(groceries):
    sales_day = alt.Chart(groceries.groupby(['day'])['item'].count().reset_index()).mark_line().encode(
        x=alt.X('day:T', title='Date'),
        y=alt.Y('item:Q', title='# of items'),
        tooltip=['day:T', 'item:Q']
    ).properties(
        width=800,
        height=300,
        title='Cummulative day transactions-2014 & 2015'
    ).configure_axis(
        grid=True
    )

    return sales_day

def three(groceries):
    #Converting weekday variable to category
    temp=groceries.copy()
    temp['qty_purchased']=groceries['id'].map(groceries['id'].value_counts())
    temp1=temp[:5000]

    #Slicing first 5000 rows as altair library can't plot any data which has record beyond that

    temp1.weekday = temp1.weekday.astype('category') 

    #Creating a new dataframe which has the frequency of weekdays
    weekday_bin=temp1['weekday'].value_counts().to_frame().reset_index().rename(columns={'index':'weekday','weekday':'count'})

    #Plotting bar chart
    bars = alt.Chart(weekday_bin).mark_bar(color="darkorange").encode(
        x='weekday',
        y=alt.Y("count",title='Number of purchases')
    )

    #Adding data labels
    text = bars.mark_text(
        align='center',
        baseline='middle',
        dy=-7 ,
        size=15,
    ).encode(
        text='count',
        tooltip=[alt.Tooltip('weekday'),
                alt.Tooltip('count')]
    )

    #Combining both
    three = (bars + text).properties(
        width=800,
        height=400,
        title="Number of quantity purchases across weekdays"
    )
    return three

def four(groceries):
    #Setting plot style
    fig = plt.figure(figsize = (15, 8))
    plt.style.use('seaborn-white')

    #Top 10 fast moving products
    plt.subplot(1,2,1)
    ax=sns.countplot(y="item", hue="year", data=groceries, palette="pastel",
                order=groceries.item.value_counts().iloc[:10].index)

    ax.set_xticklabels(ax.get_xticklabels(),fontsize=11,rotation=40, ha="right")
    ax.set_title('Top 10 Fast moving products',fontsize= 22)
    ax.set_xlabel('Total # of items purchased',fontsize = 20) 
    ax.set_ylabel('Top 10 items', fontsize = 20)
    plt.tight_layout()

    #Bottom 10 fast moving products
    plt.subplot(1,2,2)
    ax=sns.countplot(y="item", hue="year", data=groceries, palette="pastel",
                order=groceries.item.value_counts().iloc[-10:].index)
    ax.set_xticklabels(ax.get_xticklabels(),fontsize=11,rotation=40, ha="right")
    ax.set_title('Bottom 10 Fast moving products',fontsize= 22)
    ax.set_xlabel('Total # of items purchased',fontsize = 20) 
    ax.set_ylabel('Bottom 10 items', fontsize = 20)
    plt.tight_layout()
    return fig

def five(groceries):
    #Getting the top customers based on quantity purchased
    temp=groceries.copy()
    temp['qty_purchased']=groceries['id'].map(groceries['id'].value_counts())
    top_customers=temp[['id', 'qty_purchased','year']].sort_values(by = 'qty_purchased',ascending = False).head(500)

    #Converting the datatype of id and year
    top_customers.id = top_customers.id.astype('category') 
    top_customers.year = top_customers.year.astype('category') 

    #Plotting
    five = alt.Chart(top_customers).mark_bar(color="darkgreen").encode(
        x='qty_purchased',
        y=alt.Y('id', sort='-x'),
        color='year',
        tooltip=['id','qty_purchased']
    ).properties(height=400,width=600,title="Top Customers")
    return five

def six(groceries):
    #Wordcloud
    wordcloud = WordCloud(
        width = 3000,
        height = 2000,
        background_color = 'white').generate("".join(groceries['item']))
    fig = plt.figure(
        figsize = (50, 30),
        facecolor = 'k',
        edgecolor = 'k')

    #Display plot
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

    return fig

def seven(rules):
    #Setting up the style
    seven = plt.figure(figsize = (15, 15))
    plt.style.use('seaborn-white')
    #Plotting the relationship between the metrics
    plt.subplot(221)
    sns.scatterplot(x="support", y="confidence",data=rules)
    plt.subplot(222)
    sns.scatterplot(x="support", y="lift",data=rules)
    plt.subplot(223)
    sns.scatterplot(x="confidence", y="lift",data=rules)
    plt.subplot(224)
    sns.scatterplot(x="antecedent support", y="consequent support",data=rules)

    return seven



def draw_graph(rules, rules_to_show):
  
  hi = plt.figure(figsize = (15, 15))
  G1 = nx.DiGraph()
   
  color_map=[]
  N = 50
  colors = np.random.rand(N)    
  strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']   
   
   
  for i in range (rules_to_show):      
    G1.add_nodes_from(["R"+str(i)])
    
     
    for a in rules.iloc[i]['antecedents']:
                
        G1.add_nodes_from([a])
        
        G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)
       
    for c in rules.iloc[i]['consequents']:
             
        G1.add_nodes_from([a])
            
        G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)
 
  for node in G1:
    found_a_string = False
    for item in strs: 
        if node==item:
            found_a_string = True
    if found_a_string:
        color_map.append('yellow')
    else:
        color_map.append('green')       
 
 
   
  edges = list(G1.edges())
  colors = [G1[u][v]['color'] for u,v in edges]
  weights = [G1[u][v]['weight'] for u,v in edges]
 
  pos = nx.spring_layout(G1, k=16, scale=1)
  nx.draw(G1, pos, edgelist=edges, node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
  for p in pos:  # raise text positions
           pos[p][1] += 0.07
  nx.draw_networkx_labels(G1, pos)
  plt.show()

  return hi

def eight(rules):
    eight = plt.figure(figsize = (15, 15))
    rules['lhs items'] = rules['antecedents'].apply(lambda x:len(x) )
    rules[rules['lhs items']>1].sort_values('lift', ascending=False).head()

    # Replace frozen sets with strings
    rules['antecedents_'] = rules['antecedents'].apply(lambda a: ','.join(list(a)))
    rules['consequents_'] = rules['consequents'].apply(lambda a: ','.join(list(a)))

    # Transform the DataFrame of rules into a matrix using the lift metric
    pivot = rules[rules['lhs items']>1].pivot(index = 'antecedents_', 
                        columns = 'consequents_', values= 'lift')


    # Replace frozen sets with strings
    rules['antecedents_'] = rules['antecedents'].apply(lambda a: ','.join(list(a)))
    rules['consequents_'] = rules['consequents'].apply(lambda a: ','.join(list(a)))

    # Transform the DataFrame of rules into a matrix using the lift metric
    pivot = rules[rules['lhs items']>1].pivot(index = 'antecedents_', 
                        columns = 'consequents_', values= 'lift')

    # Generate a heatmap with annotations on and the colorbar off
    sns.heatmap(pivot, annot = True)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.show()
    return eight

st.title("Data Visualisations")
st.write('This visualisation charts are results of analysing the product purchase history of users.')
st.write('We initially aim to analyse the data provided from the fashion company that we are collaborating, we used mock dataset since the data from the company is confidential information.')
st.subheader("Apriori Algorithm")
groceries = csv_data('Groceries_dataset.csv')
temp=groceries.copy()
temp['qty_purchased']=groceries['id'].map(groceries['id'].value_counts())

#Creating sparse matrix 
basket = (temp.groupby(['id', 'item'])['qty_purchased']
          .sum().unstack().reset_index().fillna(0)
          .set_index('id'))

#Eoding the quantity urchased
def encode(x):
    '''Encoding the quantity of products with 0s and 1s
    0:when qty is less than or equal to 0
    1:when qty is greater than or equal to 1'''
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    
#Appying on our data
basket_sets = basket.applymap(encode)

#Apriori- Support70%
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

#Associaton rules-using lift
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

#Customizable function to change the lift and confidence
def rules_mod(lift,confidence):
    '''rules_mod is a function to control the rules 
    based on lift and confidence threshold'''
    return rules[ (rules['lift'] >= lift) &
      (rules['confidence'] >= confidence) ]


# st.dataframe(groceries.head(5))
st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <div>
            <h3>Dataset</h3>
            """
    + groceries.head(5).to_html()  # Replace df.to_html() with your actual DataFrame rendering
    + """
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write('\n')

chart1, chart2 = item_sold_each_month(groceries)

col1, col2 = st.columns(2)
st.write('Analyzes and visualizes the number of items sold each month in the years 2014 and 2015.')
st.write('\n')
st.write('\n')

with col1:
    st.altair_chart(chart1, use_container_width=False, theme="streamlit")
    
with col2:
    st.altair_chart(chart2, use_container_width=False, theme="streamlit")

two = two(groceries)
st.altair_chart(two, use_container_width=False, theme="streamlit")
st.write('Visualize cumulative day transactions for 2014 and 2015')
st.write('\n')
st.write('\n')


result = altair(groceries)
st.altair_chart(result, use_container_width=False, theme="streamlit")
st.write('Visualize purchase quantity of each month.')
st.write('\n')
st.write('\n')


three = three(groceries)
st.altair_chart(three, use_container_width=False, theme="streamlit")
st.write('This graph shows the number of quantity purchases across weekdays.')
st.write('For example, \'0\' means Monday. You can check the specific number by clicking the bar.')
st.write('\n')
st.write('\n')


# st.write('')
# four = four(groceries)
# st.pyplot(four)

st.write('')
five = five(groceries)
st.altair_chart(five, use_container_width=False, theme="streamlit")
st.write('This graph shows the top customers based on total purchase through two years.')
st.write('Sky blue represents the purchase quantity of 2014 and deep blue represents the purchase quantity of 2015.')
st.write('\n')
st.write('\n')

st.write('')
six = six(groceries)
st.pyplot(six)
st.write('This is word cloud derived from the dataset based on the frequency of words.')
st.write('The big size and eye-catching words are the products that are sold a lot.')
st.write('\n')
st.write('\n')

st.write('')
seven = seven(rules)
st.pyplot(seven)
st.write('This is scatterplots based on association rules.')
st.write('X: Support, Y: Confidence')
st.write('Support (X-axis): The proportion of transactions in the dataset that contain the items in the rule. Confidence (Y-axis): The conditional probability of the consequent given the antecedent.')
st.write('This scatterplot helps to understand how the confidence of a rule varies with its support.')

st.write('X: Support, Y: Lift')
st.write('Support (X-axis): Same as above. Lift (Y-axis): A measure of how much more likely the consequent is to occur given the antecedent, compared to its occurrence without the antecedent.')
st.write('This scatterplot helps to visualize the relationship between support and lift, indicating how much the likelihood of the consequent changes when the antecedent is present.')

st.write('X: Confidence, Y: Lift')
st.write('Confidence (X-axis): Same as above. Lift (Y-axis): Same as above.')
st.write('This scatterplot helps to explore how lift varies with confidence, giving insights into how the strength of the association (lift) changes as confidence increases.')

st.write('X: Antecedent Support, Y: Consequent Support')
st.write('Antecedent Support (X-axis): The proportion of transactions that contain the antecedent. Consequent Support (Y-axis): The proportion of transactions that contain the consequent.')
st.write('This scatterplot provides information on the individual supports of the antecedent and consequent, helping to analyze the distribution of these supports in the dataset.')
st.write('\n')
st.write('\n')

st.write('')
hi = draw_graph(rules, 10)
st.pyplot(hi)
st.write('Nodes: Each unique item or attribute (both antecedents and consequents) in the association rules becomes a node in the network graph. Nodes can represent products, features, or any other categorical variable.')
st.write('Edges: Edges connect the nodes and represent the association rules. An edge between two nodes indicates that there is a rule connecting those two items.')
st.write('Edge Attributes: Edge attributes often include information about the strength of the association (in this case lift). The color of the edge may be used to represent the strength of the association.')
st.write('\n')
st.write('\n')

st.write('')
eight = eight(rules)
st.pyplot(eight)
st.write('Each cell in the heatmap represents the lift value for the corresponding pair of items. Lift is a measure that indicates how much more likely the consequent is to occur given the antecedent compared to its occurrence without the antecedent.')