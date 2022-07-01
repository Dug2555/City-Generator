import geopy
import csv
import os
from os.path import exists
from translate import Translator
from geopy.geocoders import MapQuest

all_cities = {}

locator = MapQuest(api_key = "L61zqQuZylxWc2YinXR48VuS2NYo3tS0", user_agent="Data Mine")
Y =35.792879
X = -28.595263
while X <= 40.838327:
    Y = 35.792879
    while Y <= 71.764323:
        coordinates = Y, X
        
        location = locator.reverse(coordinates)
        place = None
        if location != None:
            #print(location.raw)
            if "adminArea5" in location.raw and location.raw["adminArea5"] != '':
                place = location.raw["adminArea5"]

                if "adminArea4" in location.raw and location.raw["adminArea4"] != '':
                    country = location.raw["adminArea4"]
                    if country not in all_cities:
                        all_cities[location.raw["adminArea4"]] = []
                    if place not in all_cities[country] and place != None:
                        all_cities[location.raw["adminArea4"]].append(place)
        Y += .1
    if exists('data.csv'):
        os.remove("data.csv")
    with open('data.csv', 'w') as f:
        writer = csv.writer(f)
        for e in all_cities.items():
            writer.writerow(e)

    print("Written " + str(X) + " and " + str(Y))
    X += .1