from django.db import models

class Attraction(models.Model):
    name = models.CharField(max_length=200)
    address = models.CharField(max_length=200, null=True, blank=True)
    rating = models.FloatField(null=True, blank=True)
    working_hours = models.CharField(max_length=50, null=True, blank=True)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)

    def __str__(self):
        return self.name

class Route(models.Model):
    start_point = models.ForeignKey(Attraction, related_name='starting_point', on_delete=models.CASCADE)
    attractions = models.ManyToManyField(Attraction, related_name='attractions')
    total_time = models.FloatField()
    total_distance = models.FloatField()

    def __str__(self):
        return f"Route from {self.start_point.name}"
