"""
Definition of models.
"""

from django.db import models

# Create your models here.

class Translate(models.Model):
	id = models.IntegerField(primary_key= True)
	Indonesia = models.CharField(db_column='Indonesia',max_length = 255)
	Sunda_Lemes = models.CharField(db_column='Sunda_Lemes',max_length = 255)
	Sunda_Sedang = models.CharField(db_column='Sunda_Sedang',max_length = 255)

	class Meta:
		managed  = False
		db_table = 'translate'