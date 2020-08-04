# -*- coding: utf-8 -*-
# Generated by Django 1.11.29 on 2020-07-07 22:23


from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('viewer', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='CmpdChoice',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('choice_type', models.CharField(choices=[(b'DE', b'Default'), (b'PR', b'Price'), (b'TO', b'Toxic')], default=b'DE', max_length=2)),
                ('score', models.FloatField(null=True)),
                ('cmpd_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='viewer.Compound')),
                ('user_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'permissions': (('view_cmpdchoice', 'View cmpdhoice'),),
            },
        ),
        migrations.CreateModel(
            name='MolAnnotation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('annotation_type', models.CharField(max_length=50)),
                ('annotation_text', models.CharField(max_length=100)),
                ('mol_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='viewer.Molecule')),
            ],
        ),
        migrations.CreateModel(
            name='MolChoice',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('choice_type', models.CharField(choices=[(b'DE', b'Default'), (b'PA', b'Pandda'), (b'GM', b'Good molecule')], default=b'DE', max_length=2)),
                ('score', models.FloatField(null=True)),
                ('mol_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='viewer.Molecule')),
                ('user_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'permissions': (('view_molchoice', 'View molchoice'),),
            },
        ),
        migrations.CreateModel(
            name='MolGroup',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('group_type', models.CharField(choices=[(b'PA', b'Pandda'), (b'DE', b'Default'), (b'MC', b'MolCluster'), (b'WC', b'WaterCluster'), (b'PC', b'PharmaCluster'), (b'RC', b'ResCluster')], default=b'DE', max_length=2)),
                ('description', models.TextField(null=True)),
                ('x_com', models.FloatField(null=True)),
                ('y_com', models.FloatField(null=True)),
                ('z_com', models.FloatField(null=True)),
                ('mol_id', models.ManyToManyField(related_name='mol_groups', to='viewer.Molecule')),
                ('target_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='viewer.Target')),
            ],
            options={
                'permissions': (('view_molgroup', 'View molecule group'),),
            },
        ),
        migrations.CreateModel(
            name='ProtChoice',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('choice_type', models.CharField(choices=[(b'DE', b'Default')], default=b'DE', max_length=2)),
                ('score', models.FloatField(null=True)),
                ('prot_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='viewer.Protein')),
                ('user_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'permissions': (('view_protchoice', 'View protchoice'),),
            },
        ),
        migrations.CreateModel(
            name='ScoreChoice',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('is_done', models.BooleanField(default=False)),
                ('choice_type', models.CharField(choices=[(b'DE', b'Default'), (b'AU', b'Docking'), (b'IT', b'Interaction fit'), (b'DF', b'Density Fit')], default=b'DE', max_length=2)),
                ('score', models.FloatField(null=True)),
                ('mol_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='viewer.Molecule')),
                ('prot_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='viewer.Protein')),
                ('user_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'permissions': (('view_scorechoice', 'View scorechoice'),),
            },
        ),
        migrations.CreateModel(
            name='ViewScene',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('uuid', models.UUIDField(unique=True)),
                ('title', models.CharField(default=b'NA', max_length=200)),
                ('scene', models.TextField()),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('modified', models.DateTimeField(auto_now=True)),
                ('snapshot', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='viewer.Snapshot')),
                ('user_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.AlterUniqueTogether(
            name='scorechoice',
            unique_together=set([('user_id', 'mol_id', 'prot_id', 'choice_type')]),
        ),
        migrations.AlterUniqueTogether(
            name='protchoice',
            unique_together=set([('user_id', 'prot_id', 'choice_type')]),
        ),
        migrations.AlterUniqueTogether(
            name='molchoice',
            unique_together=set([('user_id', 'mol_id', 'choice_type')]),
        ),
        migrations.AlterUniqueTogether(
            name='molannotation',
            unique_together=set([('mol_id', 'annotation_type')]),
        ),
        migrations.AlterUniqueTogether(
            name='cmpdchoice',
            unique_together=set([('user_id', 'cmpd_id', 'choice_type')]),
        ),
    ]
