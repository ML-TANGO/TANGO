# Generated by Django 3.2.25 on 2024-08-22 07:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tango', '0009_auto_20240822_1545'),
    ]

    operations = [
        migrations.AlterField(
            model_name='project',
            name='learning_type',
            field=models.CharField(blank=True, default='', max_length=20, null=True),
        ),
    ]
