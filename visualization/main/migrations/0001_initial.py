# pylint: disable-msg=missing-module-docstring
# pylint: disable-msg=invalid-name
from django.db import migrations, models


class Migration(migrations.Migration):
    """A dummy docstring."""

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Architecture',
            fields=[
                ('id', models.BigAutoField(auto_created=True,
                                           primary_key=True, serialize=False,
                                           verbose_name='ID')),
                ('architecture', models.FileField(upload_to='')),
            ],
        ),
        migrations.CreateModel(
            name='Edge',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False)),
                ('prior', models.IntegerField()),
                ('next', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='Node',
            fields=[
                ('order', models.IntegerField(primary_key=True,
                                              serialize=False)),
                ('layer', models.CharField(max_length=200)),
                ('parameters', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='Pth',
            fields=[
                ('id', models.BigAutoField(auto_created=True,
                                           primary_key=True, serialize=False,
                                           verbose_name='ID')),
                ('model_output', models.CharField(max_length=200)),
            ],
        ),
    ]
