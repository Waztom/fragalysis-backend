# Generated by Django 3.1 on 2022-11-08 18:45

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ActionSession',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('sessionnumber', models.IntegerField()),
                ('type', models.CharField(choices=[('reaction', 'Reaction'), ('stir', 'Stir'), ('workup', 'Workup'), ('analyse', 'Analyse')], max_length=10)),
                ('driver', models.CharField(choices=[('human', 'Human'), ('robot', 'Robot')], default='robot', max_length=10)),
                ('continuation', models.BooleanField(default=False)),
            ],
        ),
        migrations.CreateModel(
            name='Batch',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('batchtag', models.CharField(max_length=50)),
                ('batch_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='car.batch')),
            ],
        ),
        migrations.CreateModel(
            name='Column',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('index', models.IntegerField()),
                ('type', models.CharField(choices=[('reaction', 'Reaction'), ('workup1', 'Workup1'), ('workup2', 'Workup2'), ('workup3', 'Workup3'), ('spefilter', 'Spefilter'), ('lcms', 'Lcms'), ('xchem', 'Xchem'), ('nmr', 'Nmr'), ('startingmaterial', 'Startingmaterial'), ('solvent', 'Solvent')], max_length=55)),
                ('reactionclass', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Deck',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('numberslots', models.IntegerField(default=11)),
                ('slotavailable', models.BooleanField(default=True)),
                ('indexslotavailable', models.IntegerField(default=1)),
            ],
        ),
        migrations.CreateModel(
            name='Method',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nosteps', models.IntegerField()),
                ('otchem', models.BooleanField(default=False)),
            ],
        ),
        migrations.CreateModel(
            name='OTBatchProtocol',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('celery_taskid', models.CharField(max_length=50)),
                ('zipfile', models.FileField(max_length=255, null=True, upload_to='otbatchprotocols/')),
                ('batch_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='otbatchprotocols', to='car.batch')),
            ],
        ),
        migrations.CreateModel(
            name='OTSession',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('reactionstep', models.IntegerField()),
                ('sessiontype', models.CharField(choices=[('reaction', 'Reaction'), ('workup', 'Workup'), ('lcmsprep', 'Lcmsprep')], default='reaction', max_length=10)),
                ('otbatchprotocol_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='otsessions', to='car.otbatchprotocol')),
            ],
        ),
        migrations.CreateModel(
            name='Plate',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('labware', models.CharField(max_length=255)),
                ('index', models.IntegerField()),
                ('name', models.CharField(max_length=255, null=True)),
                ('type', models.CharField(choices=[('reaction', 'Reaction'), ('workup1', 'Workup1'), ('workup2', 'Workup2'), ('workup3', 'Workup3'), ('spefilter', 'Spefilter'), ('lcms', 'Lcms'), ('xchem', 'Xchem'), ('nmr', 'Nmr'), ('startingmaterial', 'Startingmaterial'), ('solvent', 'Solvent')], max_length=55, null=True)),
                ('maxwellvolume', models.FloatField()),
                ('numberwells', models.IntegerField()),
                ('wellavailable', models.BooleanField(default=True)),
                ('numberwellsincolumn', models.IntegerField(default=8)),
                ('indexswellavailable', models.IntegerField(default=0)),
                ('numbercolumns', models.IntegerField()),
                ('columnavailable', models.BooleanField(default=True)),
                ('indexcolumnavailable', models.IntegerField(default=0)),
                ('deck_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='car.deck')),
                ('otbatchprotocol_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='otplates', to='car.otbatchprotocol')),
                ('otsession_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='car.otsession')),
            ],
        ),
        migrations.CreateModel(
            name='Project',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('init_date', models.DateTimeField(auto_now_add=True)),
                ('name', models.SlugField(max_length=100)),
                ('submitterorganisation', models.CharField(max_length=100)),
                ('submittername', models.CharField(max_length=255)),
                ('proteintarget', models.CharField(max_length=100)),
                ('quotedcost', models.FloatField(null=True)),
                ('quoteurl', models.CharField(max_length=255, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='PubChemInfo',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('compoundid', models.IntegerField()),
                ('summaryurl', models.CharField(max_length=255)),
                ('lcssurl', models.CharField(max_length=255)),
                ('smiles', models.CharField(max_length=255)),
                ('cas', models.CharField(max_length=50, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Reaction',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('reactionclass', models.CharField(max_length=255)),
                ('number', models.IntegerField()),
                ('intramolecular', models.BooleanField(default=False)),
                ('recipetype', models.CharField(default='standard', max_length=50, null=True)),
                ('temperature', models.IntegerField(default=25)),
                ('image', models.FileField(max_length=255, null=True, upload_to='reactionimages/')),
                ('success', models.BooleanField(default=True)),
                ('method_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='reactions', to='car.method')),
            ],
        ),
        migrations.CreateModel(
            name='Well',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('clonewellid', models.IntegerField(null=True)),
                ('index', models.IntegerField()),
                ('type', models.CharField(choices=[('reaction', 'Reaction'), ('workup1', 'Workup1'), ('workup2', 'Workup2'), ('workup3', 'Workup3'), ('spefilter', 'Spefilter'), ('lcms', 'Lcms'), ('xchem', 'Xchem'), ('nmr', 'Nmr'), ('startingmaterial', 'Startingmaterial'), ('solvent', 'Solvent')], max_length=55)),
                ('volume', models.FloatField(null=True)),
                ('smiles', models.CharField(max_length=255, null=True)),
                ('concentration', models.FloatField(null=True)),
                ('solvent', models.CharField(max_length=255, null=True)),
                ('reactantfornextstep', models.BooleanField(default=False)),
                ('available', models.BooleanField(default=True)),
                ('column_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='car.column')),
                ('method_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='car.method')),
                ('otsession_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='otwells', to='car.otsession')),
                ('plate_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='car.plate')),
                ('reaction_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='car.reaction')),
            ],
        ),
        migrations.CreateModel(
            name='TipRack',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('labware', models.CharField(max_length=255)),
                ('index', models.IntegerField()),
                ('name', models.CharField(max_length=255)),
                ('deck_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='car.deck')),
                ('otsession_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='car.otsession')),
            ],
        ),
        migrations.CreateModel(
            name='Target',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('smiles', models.CharField(db_index=True, max_length=255)),
                ('image', models.FileField(max_length=255, upload_to='targetimages/')),
                ('name', models.CharField(db_index=True, max_length=255)),
                ('mass', models.FloatField()),
                ('mols', models.FloatField()),
                ('batch_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='targets', to='car.batch')),
            ],
        ),
        migrations.CreateModel(
            name='StirAction',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('number', models.IntegerField()),
                ('platetype', models.CharField(choices=[('reaction', 'Reaction'), ('workup1', 'Workup1'), ('workup2', 'Workup2'), ('workup3', 'Workup3'), ('spefilter', 'Spefilter'), ('lcms', 'Lcms'), ('xchem', 'Xchem'), ('nmr', 'Nmr'), ('startingmaterial', 'Startingmaterial'), ('solvent', 'Solvent')], max_length=20)),
                ('duration', models.FloatField()),
                ('durationunit', models.CharField(choices=[('seconds', 'Seconds'), ('minutes', 'Minutes'), ('hours', 'Hours')], default='hours', max_length=10)),
                ('temperature', models.IntegerField()),
                ('temperatureunit', models.CharField(choices=[('degC', 'Degcel'), ('K', 'Kelvin')], default='degC', max_length=10)),
                ('stirringspeed', models.CharField(choices=[('gentle', 'Gentle'), ('normal', 'Normal'), ('vigorous', 'Vigorous')], default='normal', max_length=10)),
                ('actionsession_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='stiractions', to='car.actionsession')),
                ('reaction_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='stiractions', to='car.reaction')),
            ],
        ),
        migrations.CreateModel(
            name='SolventPrep',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('solventprepcsv', models.FileField(max_length=255, upload_to='solventprep/')),
                ('otsession_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='solventpreps', to='car.otsession')),
            ],
        ),
        migrations.CreateModel(
            name='Reactant',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('smiles', models.CharField(max_length=255)),
                ('previousreactionproduct', models.BooleanField(default=False)),
                ('pubcheminfo_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.PROTECT, related_name='reactantpubcheminfo', to='car.pubcheminfo')),
                ('reaction_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='reactants', to='car.reaction')),
            ],
        ),
        migrations.CreateModel(
            name='Product',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('smiles', models.CharField(db_index=True, max_length=255, null=True)),
                ('image', models.FileField(max_length=255, upload_to='productimages/')),
                ('pubcheminfo_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.PROTECT, related_name='productpubcheminfo', to='car.pubcheminfo')),
                ('reaction_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='products', to='car.reaction')),
            ],
        ),
        migrations.CreateModel(
            name='Pipette',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('position', models.CharField(choices=[('Right', 'Right'), ('Left', 'Left')], max_length=10)),
                ('labware', models.CharField(max_length=100)),
                ('type', models.CharField(max_length=255)),
                ('maxvolume', models.FloatField(default=300)),
                ('name', models.CharField(max_length=255)),
                ('otsession_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='pipettes', to='car.otsession')),
            ],
        ),
        migrations.CreateModel(
            name='OTScript',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('otscript', models.FileField(max_length=255, upload_to='otscripts/')),
                ('otsession_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='otscripts', to='car.otsession')),
            ],
        ),
        migrations.CreateModel(
            name='OTProject',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('init_date', models.DateTimeField(auto_now_add=True)),
                ('name', models.CharField(max_length=150)),
                ('project_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='car.project')),
            ],
        ),
        migrations.AddField(
            model_name='otbatchprotocol',
            name='otproject_id',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='car.otproject'),
        ),
        migrations.CreateModel(
            name='MixAction',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('number', models.IntegerField()),
                ('platetype', models.CharField(choices=[('reaction', 'Reaction'), ('workup1', 'Workup1'), ('workup2', 'Workup2'), ('workup3', 'Workup3'), ('spefilter', 'Spefilter'), ('lcms', 'Lcms'), ('xchem', 'Xchem'), ('nmr', 'Nmr'), ('startingmaterial', 'Startingmaterial'), ('solvent', 'Solvent')], max_length=20)),
                ('repetitions', models.IntegerField()),
                ('calcunit', models.CharField(choices=[('masseq', 'Masseq'), ('ul', 'Ul')], default='moleq', max_length=10)),
                ('volume', models.FloatField()),
                ('volumeunit', models.CharField(choices=[('ul', 'Ul'), ('ml', 'Ml')], default='ul', max_length=2)),
                ('actionsession_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='car.actionsession')),
                ('reaction_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='mixactions', to='car.reaction')),
            ],
        ),
        migrations.AddField(
            model_name='method',
            name='target_id',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='methods', to='car.target'),
        ),
        migrations.CreateModel(
            name='ExtractAction',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('number', models.IntegerField()),
                ('fromplatetype', models.CharField(choices=[('reaction', 'Reaction'), ('workup1', 'Workup1'), ('workup2', 'Workup2'), ('workup3', 'Workup3'), ('spefilter', 'Spefilter'), ('lcms', 'Lcms'), ('xchem', 'Xchem'), ('nmr', 'Nmr'), ('startingmaterial', 'Startingmaterial'), ('solvent', 'Solvent')], max_length=20)),
                ('toplatetype', models.CharField(choices=[('reaction', 'Reaction'), ('workup1', 'Workup1'), ('workup2', 'Workup2'), ('workup3', 'Workup3'), ('spefilter', 'Spefilter'), ('lcms', 'Lcms'), ('xchem', 'Xchem'), ('nmr', 'Nmr'), ('startingmaterial', 'Startingmaterial'), ('solvent', 'Solvent')], max_length=20)),
                ('layer', models.CharField(choices=[('top', 'Top'), ('bottom', 'Bottom')], default='bottom', max_length=10)),
                ('smiles', models.CharField(max_length=255)),
                ('volume', models.FloatField()),
                ('volumeunit', models.CharField(default='ul', max_length=2)),
                ('molecularweight', models.FloatField()),
                ('bottomlayervolume', models.FloatField(null=True)),
                ('bottomlayervolumeunit', models.CharField(default='ul', max_length=2)),
                ('solvent', models.CharField(max_length=255, null=True)),
                ('concentration', models.FloatField(null=True)),
                ('actionsession_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='car.actionsession')),
                ('reaction_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='extractactions', to='car.reaction')),
            ],
        ),
        migrations.AddField(
            model_name='deck',
            name='otsession_id',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='otdecks', to='car.otsession'),
        ),
        migrations.CreateModel(
            name='CompoundOrder',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ordercsv', models.FileField(max_length=255, upload_to='compoundorders/')),
                ('otsession_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='compoundorders', to='car.otsession')),
            ],
        ),
        migrations.AddField(
            model_name='column',
            name='otsession_id',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='otcolumns', to='car.otsession'),
        ),
        migrations.AddField(
            model_name='column',
            name='plate_id',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='car.plate'),
        ),
        migrations.CreateModel(
            name='CatalogEntry',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('vendor', models.CharField(max_length=100)),
                ('catalogid', models.CharField(max_length=50)),
                ('priceinfo', models.CharField(max_length=50)),
                ('upperprice', models.IntegerField(null=True)),
                ('leadtime', models.IntegerField(null=True)),
                ('reactant_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='catalogentries', to='car.reactant')),
                ('target_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='catalogentries', to='car.target')),
            ],
        ),
        migrations.AddField(
            model_name='batch',
            name='project_id',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='batches', to='car.project'),
        ),
        migrations.CreateModel(
            name='AddAction',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('number', models.IntegerField()),
                ('fromplatetype', models.CharField(choices=[('reaction', 'Reaction'), ('workup1', 'Workup1'), ('workup2', 'Workup2'), ('workup3', 'Workup3'), ('spefilter', 'Spefilter'), ('lcms', 'Lcms'), ('xchem', 'Xchem'), ('nmr', 'Nmr'), ('startingmaterial', 'Startingmaterial'), ('solvent', 'Solvent')], max_length=20)),
                ('toplatetype', models.CharField(choices=[('reaction', 'Reaction'), ('workup1', 'Workup1'), ('workup2', 'Workup2'), ('workup3', 'Workup3'), ('spefilter', 'Spefilter'), ('lcms', 'Lcms'), ('xchem', 'Xchem'), ('nmr', 'Nmr'), ('startingmaterial', 'Startingmaterial'), ('solvent', 'Solvent')], max_length=20)),
                ('smiles', models.CharField(max_length=255)),
                ('calcunit', models.CharField(choices=[('moleq', 'Moleq'), ('masseq', 'Masseq'), ('ul', 'Ul')], default='moleq', max_length=10)),
                ('volume', models.FloatField()),
                ('volumeunit', models.CharField(choices=[('ul', 'Ul'), ('ml', 'Ml')], default='ul', max_length=2)),
                ('molecularweight', models.FloatField()),
                ('solvent', models.CharField(max_length=255, null=True)),
                ('concentration', models.FloatField(null=True)),
                ('actionsession_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='car.actionsession')),
                ('reaction_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='addactions', to='car.reaction')),
            ],
        ),
        migrations.AddField(
            model_name='actionsession',
            name='reaction_id',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='actionsessions', to='car.reaction'),
        ),
    ]
