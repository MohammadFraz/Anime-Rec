from django.db import models


class Genre(models.Model):
    gid = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=20)

    def __str__(self):
        return str(self.gid) + ': ' + self.name


class Anime(models.Model):
    aid = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=100)
    rating = models.DecimalField(max_digits=4, decimal_places=2)
    related = models.ManyToManyField('self')
    genre = models.ManyToManyField(Genre)
    members = models.IntegerField()

    def __str__(self):
        return str(self.aid) + ': ' + self.name

    @property
    def genre_list(self):
        return ', '.join([g.name for g in self.genre.all()])
