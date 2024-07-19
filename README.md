# Rating-Prediction-Zomato-Dataset

## Dataset Overview

The dataset used for this project is sourced from Zomato, a popular restaurant discovery and food delivery platform. The dataset includes various features that provide detailed information about the restaurants, such as customer votes, cost estimates, and other attributes. The aim of this project is to predict the ratings of restaurants based on these features.

### Features Description

#### Numerical Features
1. **votes**: 
   - **Description**: The number of votes a restaurant has received from customers. This feature reflects the popularity and customer engagement of the restaurant.
   - **Example**: 250, 1000

2. **approx_cost(for two people)**: 
   - **Description**: The approximate cost for two people to dine at the restaurant. This feature provides an insight into the pricing level of the restaurant.
   - **Example**: 500, 1500

#### Categorical Features
1. **online_order**: 
   - **Description**: Indicates whether the restaurant offers online ordering.
   - **Values**: 'Yes', 'No'
   - **Example**: Yes

2. **book_table**: 
   - **Description**: Indicates whether the restaurant allows table booking.
   - **Values**: 'Yes', 'No'
   - **Example**: No

3. **rest_type**: 
   - **Description**: The type of restaurant, indicating the dining style and service model.
   - **Example**: 'Casual Dining', 'Cafe', 'Quick Bites'

4. **listed_in(type)**: 
   - **Description**: The type or category of the restaurant listing.
   - **Example**: 'Buffet', 'Delivery', 'Drinks & nightlife'

5. **listed_in(city)**: 
   - **Description**: The city where the restaurant is located. This feature helps in understanding the geographical distribution and regional preferences.
   - **Example**: 'Bangalore', 'Mumbai'

#### Text Feature
1. **cuisines**: 
   - **Description**: The cuisines offered by the restaurant. This feature provides a variety of food options available at the restaurant.
   - **Example**: 'Italian, Chinese', 'North Indian, Mughlai'