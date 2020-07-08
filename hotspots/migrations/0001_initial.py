# -*- coding: utf-8 -*-
# Generated by Django 1.11.29 on 2020-07-07 22:23
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('viewer', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='HotspotMap',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('map_type', models.CharField(choices=[(b'AC', b'Acceptor'), (b'AP', b'Apolar'), (b'DO', b'Donor')], max_length=2)),
                ('map_info', models.FileField(max_length=255, null=True, upload_to=b'maps/')),
                ('compressed_map_info', models.FileField(max_length=255, null=True, upload_to=b'maps/')),
                ('prot_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='viewer.Protein')),
                ('target_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='viewer.Target')),
            ],
        ),
        migrations.AlterUniqueTogether(
            name='hotspotmap',
            unique_together=set([('map_type', 'target_id', 'prot_id')]),
        ),
    ]
