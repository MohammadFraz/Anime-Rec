from django.conf.urls import url
from recommender import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^recommender', views.recommender, name='recommender'),
    url(r'^credits', views.credits, name='credits')
]