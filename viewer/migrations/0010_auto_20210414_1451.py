# Generated by Django 3.1 on 2021-04-14 14:51

import django.core.serializers.json
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('viewer', '0009_moleculetag_sessionprojecttag_tagcategory'),
    ]

    operations = [
        migrations.AlterField(
            model_name='moleculetag',
            name='additional_info',
            field=models.JSONField(default='', encoder=django.core.serializers.json.DjangoJSONEncoder),
        ),
        migrations.AlterField(
            model_name='sessionprojecttag',
            name='additional_info',
            field=models.JSONField(default='', encoder=django.core.serializers.json.DjangoJSONEncoder),
        ),
    ]
