# What is the Research Question and how is it related to the theme of "Understanding the liveability, inclusiveness, health and sustainability of communities in Victoria"?

    Our research questions is:
        Does the average income of a Suburb / Statistical Area affect their access to the public transport network?

    The question relates to the themes of liveability and inclusiveness of all Melbourne suburbs, as connection to the public transport network is a neccesity for commuting, social lives and the access to public facility's made available by the Government.



# Why is it worth tackling (i.e. motivation) and who would care? Who are the stakeholders in this project? In what respects might it provide innovative information? (you do not want to have a question which is trivial, or for which the answer already publicly exists and can be readily found)?

    We believe this task is worth tackling as it can provide a great basis of information to show which of our suburbs are lacking in PTV infrastructure, and can also increase awareness on what areas in our city are better connected and suited to people who rely moreso on the transport network.

    A stakeholders in our project would be the Victorian Government / PTV so they can better know where the network is lacking, and if and where they should devote more resources in order to better serve the public transport sector.
    A second stakeholder would be Yarra Trams, Metro Trains and the multitude of bus operators so that they can have better knowledge on where to expand their network and where to deliver more services.
    Lastly and probably the primary stakeholder, would be the people of Victoria who rely on the transport network for their day to day lives. An increase in awareness to where is better connected, and to where infrastructure upgrades are coming can help people decide which locations better suit them for their commuting needs.




# What are the two or more open datasets you could use and why? What is their format, size and what information do they contain? Can they be linked together, and if so how?
    



ABS - Data by Region - Family and Community, ASGS, 2011 to 2018 
- 15976 rows by 126 columns
- Format : Excel Spreadsheet 
- It contains information indexed using SA2 - SA4 regions throughout Australia. Columns contain mode of transport to work, houshold types, family types, social status etc. 

ABS - Data by Region - Income (including Government Allowances), ASGS, 2011 to 2018
- 18691 rows by 72 columns 
- Format : Excel Spreadsheet
- It contains income information (etc median, mean, investment income etc.) indexed using SA2 - SA4 regions throughout Australia. 

We can definitely group the above two data sets by location as they use the same index. We will have to decide upon using SA2, SA3, or SA4 area, and will pick the grouping that conincides with out PTV tram,train,bus, bicycle storage data. 

PTV - Metro Tram Stops (Points)
- 1666 rows by 7 columns 
- Format: csv
-  Contains stop ID, stop name(relavant to location), and other atrributes such as ObjectID, routes that use that stop etc.

PTV - Metro Train Stations (Points)
- 221 rows by 7 columns 
- Format: csv
- Contains info about STOP_ID and routes using the stop. 

PTV - Metro Bus Stops (Points)
- 150001 rows by 7 columns
- Format: csv
- Contains stop ID, stop name(relavant to location), and other atrributes such as ObjectID, routes that use that stop etc.


PTV - Train Station Bike Storage (Points)
- 771 rows by 5 columns 
- Format: csv
- Contains information regarding the number of bicycle storage units at train stations. 


We plan on grouping all data sets by location. We will convert the csv files to excel or vice versa.





# What data wrangling and analysis methodologies could you use to investigate your research question? Be specific about how they could be applied on your selected datasets.

Grouping ABS data by location will be simple as it is used as an index. Then we can select the year which most appropriately coincides with the PTV data. From that point we can collect information such as mean income, how many people travel to work via bicycle etc..

Grouping the PTV datasets may prove to be more difficult as the location is embedded within the name, and not all names specify a region. We will have to use regular expressions to find exactly what we are looking for. We also may have to refer to another dataset that links a surburb to a region (This is an example, as we havent decided on how to group areas). 

In terms of analysis, we will attempt to find correlation, mutual information, and other diagnostic statistics between income statistics, and the amount of PTV infrastructure to assess the livability and inclusiveness amongst income groups of the Victorian Metro regarding public transport infrastructure.



# What could be achieved by using these data wrangling methodologies? What would be output/product of the wrangling (type of data, graph, table, statistic(s), ...)? How will this add value compared to having just the raw data?

By combining the separate datasets, we are aiming to gain a better insight into the relationship between the average income of a suburb/SA and availability/access to public transport. 

We are hoping to obtain tabulated data such as mean income and access to PTV infrastructure through data wrangling which will be grouped by location. We will also  represent data graphically which will be easier to understand at a glance for the general public which are our primary stakeholders. 


# What might be the challenges and risks for undertaking this work?

In terms of challenges, the PTV data points are large in quantity and do not share the same index as the ABS data hence grouping them might be difficult. Another challenge is to present the information in a way where it can be easily understood by the general population.
