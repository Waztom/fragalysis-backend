# -*- coding: utf-8 -*-
# Generated by Django 1.11.29 on 2020-07-07 22:28


from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('viewer', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='target',
            name='zip_archive',
            field=models.FileField(max_length=255, null=True, upload_to=b'archive/'),
        ),
    ]
