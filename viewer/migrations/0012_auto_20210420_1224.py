# Generated by Django 3.1 on 2021-04-20 12:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('viewer', '0011_auto_20210420_1211'),
    ]

    operations = [
        migrations.AlterField(
            model_name='moleculetag',
            name='tag',
            field=models.CharField(max_length=200),
        ),
        migrations.AlterField(
            model_name='sessionprojecttag',
            name='tag',
            field=models.CharField(max_length=200),
        ),
        migrations.AlterUniqueTogether(
            name='moleculetag',
            unique_together={('tag', 'target')},
        ),
        migrations.AlterUniqueTogether(
            name='sessionprojecttag',
            unique_together={('tag', 'target')},
        ),
    ]
