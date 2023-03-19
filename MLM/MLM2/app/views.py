from django.shortcuts import render
import fastai
from fastai import *
from fastai.text import *
from fastai.callback import *
import transformers
# Create your views here.
@csrf_exempt
@api_view(('POST',))
@action(detail=False, methods=['POST'])
def Scrapping(request):
    if request.method == 'POST':
        import requests
        from bs4 import BeautifulSoup
        import imdb

        # create an instance of the IMDb class
        ia = imdb.IMDb()
        movie_title="Inception"
        # search for the movie by title
        results = ia.search_movie(movie_title)

        # get the first result from the search
        movie = results[0]

        # print the IMDb ID of the movie
        print(movie.getID())
        # Define the URL of the IMDb movie page you want to scrape
        url = f"https://www.imdb.com/title/tt{movie.getID()}/reviews"

        # Send a request to the website and get the HTML content
        response = requests.get(url)
        html_content = response.content

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")

        # Find all the review containers on the page
        review_containers = soup.find_all("div", class_="lister-item-content")

        # Set a variable to keep track of the number of reviews
        count = 0
        l=[]
        # Loop through each review container and extract the relevant information
        for container in review_containers:
            # Exit the loop if we have scraped 10 reviews
            if count == 10:
                break
            
            # Extract the review text
            review_text = container.find("div", class_="text").get_text().strip()
            
            # Extract the review rating
            review_rating = container.find("span", class_="rating-other-user-rating").find("span").get_text()
            
            # Print the review text and rating
            print("Review text: ", review_text)
            l.append(review_text)
            print("Review rating: ", review_rating)
            
            # Increment the count of reviews
            count += 1
        
    return count