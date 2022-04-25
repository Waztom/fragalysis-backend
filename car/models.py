from django.db import models


class Project(models.Model):
    init_date = models.DateTimeField(auto_now_add=True)
    name = models.SlugField(max_length=100, db_index=True)
    submitterorganisation = models.CharField(max_length=100)
    submittername = models.CharField(max_length=255)
    proteintarget = models.CharField(max_length=100)
    quotedcost = models.FloatField(null=True)
    quoteurl = models.CharField(max_length=255, null=True)


class Batch(models.Model):
    project_id = models.ForeignKey(
        Project, related_name="batches", on_delete=models.CASCADE
    )
    batch_id = models.ForeignKey("Batch", on_delete=models.CASCADE, null=True)
    batch_tag = models.CharField(max_length=50)


class Target(models.Model):
    class Unit(models.TextChoices):
        mmol = "mmol"
        g = "g"
        mg = "mg"

    batch_id = models.ForeignKey(
        Batch, related_name="targets", on_delete=models.CASCADE
    )
    smiles = models.CharField(max_length=255, db_index=True, null=True)
    image = models.FileField(upload_to="targetimages/", max_length=255)
    name = models.CharField(max_length=255, db_index=True)
    targetmass = models.FloatField()
    targetmols = models.FloatField()
    unit = models.CharField(choices=Unit.choices, default=Unit.mg, max_length=10)


class Method(models.Model):
    target_id = models.ForeignKey(
        Target, related_name="methods", on_delete=models.CASCADE
    )
    status = models.BooleanField(default=True)
    nosteps = models.IntegerField(null=True)
    estimatecost = models.FloatField(default=100)
    synthesise = models.BooleanField(default=True)
    otchem = models.BooleanField(default=False)


class Reaction(models.Model):
    method_id = models.ForeignKey(
        Method, related_name="reactions", on_delete=models.CASCADE
    )
    reactionclass = models.CharField(max_length=255)
    reactiontemperature = models.IntegerField(default=25)
    reactionimage = models.FileField(
        upload_to="reactionimages/",
        max_length=255,
        null=True,
    )
    successrate = models.FloatField(default=0.5)
    success = models.BooleanField(default=True)


class Reactant(models.Model):
    reaction_id = models.ForeignKey(
        Reaction, related_name="reactants", on_delete=models.CASCADE, null=True
    )
    smiles = models.CharField(max_length=255)


class CatalogEntry(models.Model):
    reactant_id = models.ForeignKey(
        Reactant, related_name="catalogentries", on_delete=models.CASCADE, null=True
    )
    target_id = models.ForeignKey(
        Target, related_name="catalogentries", on_delete=models.CASCADE, null=True
    )

    vendor = models.CharField(max_length=100)
    catalogid = models.CharField(max_length=50)
    priceinfo = models.CharField(max_length=50)
    upperprice = models.IntegerField(null=True)
    leadtime = models.IntegerField(null=True)


class Product(models.Model):
    reaction_id = models.ForeignKey(
        Reaction, related_name="products", on_delete=models.CASCADE
    )
    name = models.CharField(max_length=255)
    smiles = models.CharField(max_length=255, db_index=True, null=True)
    image = models.FileField(upload_to="productimages/", max_length=255)
    mculeid = models.CharField(max_length=255, null=True)


class AnalyseAction(models.Model):
    class QCMethod(models.TextChoices):
        LCMS = "LCMS"
        NMR = "NMR"
        XChem = "XChem"

    reaction_id = models.ForeignKey(Reaction, on_delete=models.CASCADE)
    actiontype = models.CharField(max_length=100)
    actionno = models.IntegerField()
    method = models.CharField(
        choices=QCMethod.choices, default=QCMethod.LCMS, max_length=10
    )


class AddAction(models.Model):
    class Unit(models.TextChoices):
        mmol = "mmol"
        ml = "ml"
        ul = "ul"
        moleq = "moleq"

    class Atmosphere(models.TextChoices):
        nitrogen = "nitrogen"
        air = "air"

    reaction_id = models.ForeignKey(
        Reaction, related_name="addactions", on_delete=models.CASCADE
    )
    actiontype = models.CharField(max_length=100)
    actionno = models.IntegerField()
    additionorder = models.IntegerField(null=True)
    material = models.CharField(max_length=255, null=True)
    materialsmiles = models.CharField(max_length=255, null=True)
    materialquantity = models.FloatField()
    materialquantityunit = models.CharField(
        choices=Unit.choices,
        default=Unit.ul,
        max_length=10,
    )

    dropwise = models.BooleanField(default=False)
    atmosphere = models.CharField(
        choices=Atmosphere.choices, default=Atmosphere.air, max_length=10
    )

    # These are extras for helping robotic execution/calcs
    molecularweight = models.FloatField(null=True)
    materialimage = models.FileField(
        upload_to="addactionimages/",
        max_length=255,
        null=True,
    )
    solvent = models.CharField(max_length=255, null=True)
    concentration = models.FloatField(null=True, blank=True)
    mculeid = models.CharField(max_length=255, null=True)
    mculeprice = models.FloatField(null=True)
    mculeurl = models.CharField(max_length=255, null=True)
    mculedeliverytime = models.IntegerField(null=True)


class ExtractAction(models.Model):
    class Unit(models.TextChoices):
        ml = "ml"
        mmol = "mmol"
        moleq = "moleq"

    reaction_id = models.ForeignKey(
        Reaction, related_name="extractactions", on_delete=models.CASCADE
    )
    actiontype = models.CharField(max_length=100)
    actionno = models.IntegerField()
    solvent = models.CharField(max_length=100)
    solventquantity = models.FloatField(null=True)
    solventquantityunit = models.CharField(
        choices=Unit.choices, default=Unit.ml, max_length=10
    )
    numberofrepetitions = models.IntegerField(null=True)


class FilterAction(models.Model):
    class PhaseToKeep(models.TextChoices):
        filtrate = "filtrate"
        precipitate = "precipitate"

    class Unit(models.TextChoices):
        ml = "ml"
        mmol = "mmol"
        moleq = "moleq"

    reaction_id = models.ForeignKey(
        Reaction, related_name="filteractions", on_delete=models.CASCADE
    )
    actiontype = models.CharField(max_length=100)
    actionno = models.IntegerField()
    phasetokeep = models.CharField(
        choices=PhaseToKeep.choices, default=PhaseToKeep.filtrate, max_length=20
    )
    rinsingsolvent = models.CharField(max_length=255, null=True)
    rinsingsolventquantity = models.IntegerField(null=True)
    rinsingsolventquantityunit = models.CharField(
        choices=Unit.choices,
        default=Unit.ml,
        max_length=10,
    )

    extractionforprecipitatesolvent = models.CharField(max_length=255, null=True)
    extractionforprecipitatesolventquantity = models.IntegerField(null=True)
    extractionforprecipitatesolventquantityunit = models.CharField(
        choices=Unit.choices,
        default=Unit.ml,
        max_length=10,
    )


class QuenchAction(models.Model):
    class TemperatureUnit(models.TextChoices):
        degcel = "degC"
        kelvin = "K"

    class Unit(models.TextChoices):
        ml = "ml"
        mmol = "mmol"
        moleq = "moleq"

    reaction_id = models.ForeignKey(
        Reaction, related_name="quenchactions", on_delete=models.CASCADE
    )
    actiontype = models.CharField(max_length=100)
    actionno = models.IntegerField()

    material = models.CharField(max_length=255)
    materialquantity = models.FloatField(null=True)
    materialquantityunit = models.CharField(
        choices=Unit.choices, default=Unit.ml, max_length=10
    )
    dropwise = models.BooleanField(default=False)
    temperature = models.IntegerField(null=True)
    temperatureunit = models.CharField(
        choices=TemperatureUnit.choices, default=TemperatureUnit.degcel, max_length=10
    )


class SetTemperatureAction(models.Model):
    class TemperatureUnit(models.TextChoices):
        degcel = "degC"
        kelvin = "K"

    reaction_id = models.ForeignKey(
        Reaction, related_name="settemperatureactions", on_delete=models.CASCADE
    )
    actiontype = models.CharField(max_length=100)
    actionno = models.IntegerField()
    temperature = models.IntegerField()
    temperatureunit = models.CharField(
        choices=TemperatureUnit.choices, default=TemperatureUnit.degcel, max_length=10
    )


class StirAction(models.Model):
    class TemperatureUnit(models.TextChoices):
        degcel = "degC"
        kelvin = "K"

    class Unit(models.TextChoices):
        seconds = "seconds"
        minutes = "minutes"
        hours = "hours"

    class Speed(models.TextChoices):
        gentle = "gentle"
        normal = "normal"
        vigorous = "vigorous"

    class Atmosphere(models.TextChoices):
        nitrogen = "nitrogen"
        air = "air"

    reaction_id = models.ForeignKey(
        Reaction, related_name="stiractions", on_delete=models.CASCADE
    )
    actiontype = models.CharField(max_length=100)
    actionno = models.IntegerField()
    duration = models.FloatField(null=True)
    durationunit = models.CharField(
        choices=Unit.choices, default=Unit.hours, max_length=10
    )
    temperature = models.IntegerField(null=True)
    temperatureunit = models.CharField(
        choices=TemperatureUnit.choices, default=TemperatureUnit.degcel, max_length=10
    )
    stirringspeed = models.CharField(
        choices=Speed.choices, default=Speed.normal, max_length=10
    )
    atmosphere = models.CharField(
        choices=Atmosphere.choices, default=Atmosphere.air, max_length=10
    )


# Mcule models
class MculeQuote(models.Model):
    project_id = models.ForeignKey(Project, on_delete=models.CASCADE)
    quoteid = models.CharField(max_length=255)
    quoteurl = models.CharField(max_length=255)
    quotecost = models.FloatField()
    quotevaliduntil = models.CharField(max_length=255)


class MCuleOrder(models.Model):
    project_id = models.ForeignKey(Project, on_delete=models.CASCADE)
    orderplatecsv = models.FileField(upload_to="mculeorderplates/", max_length=255)


# Models for capturing OT session, Deck, Plates and Wells
class OTProtocol(models.Model):
    project_id = models.ForeignKey(Project, on_delete=models.CASCADE)
    init_date = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=150)


class OTBatchProtocol(models.Model):
    otprotocol_id = models.ForeignKey(OTProtocol, on_delete=models.CASCADE)
    batch_id = models.ForeignKey(
        Batch, related_name="otbatchprotocols", on_delete=models.CASCADE
    )
    celery_task_id = models.CharField(max_length=50)
    zipfile = models.FileField(upload_to="otbatchprotocols/", max_length=255, null=True)


class OTSession(models.Model):
    otbatchprotocol_id = models.ForeignKey(
        OTBatchProtocol, related_name="otsessions", on_delete=models.CASCADE
    )
    init_date = models.DateTimeField(auto_now_add=True)
    reactionstep = models.IntegerField()


class Deck(models.Model):
    otsession_id = models.ForeignKey(
        OTSession, related_name="decks", on_delete=models.CASCADE
    )
    numberslots = models.IntegerField(default=11)
    slotavailable = models.BooleanField(default=True)
    indexslotavailable = models.IntegerField(default=1)


class Pipette(models.Model):
    class Position(models.TextChoices):
        right = "Right"
        left = "Left"

    otsession_id = models.ForeignKey(
        OTSession, related_name="pipettes", on_delete=models.CASCADE
    )
    pipettename = models.CharField(max_length=255)
    labware = models.CharField(max_length=100)
    type = models.CharField(max_length=255)
    position = models.CharField(choices=Position.choices, max_length=10)
    maxvolume = models.FloatField(default=300)


class TipRack(models.Model):
    deck_id = models.ForeignKey(Deck, on_delete=models.CASCADE)
    otsession_id = models.ForeignKey(OTSession, on_delete=models.CASCADE)
    tiprackname = models.CharField(max_length=255)
    tiprackindex = models.IntegerField()
    labware = models.CharField(max_length=255)


class Plate(models.Model):
    deck_id = models.ForeignKey(Deck, on_delete=models.CASCADE)
    otsession_id = models.ForeignKey(OTSession, on_delete=models.CASCADE)
    platename = models.CharField(max_length=255)
    plateindex = models.IntegerField()
    labware = models.CharField(max_length=255)
    maxwellvolume = models.FloatField()
    numberwells = models.IntegerField()
    wellavailable = models.BooleanField(default=True)
    indexswellavailable = models.IntegerField(default=0)


class Well(models.Model):
    plate_id = models.ForeignKey(Plate, on_delete=models.CASCADE)
    method_id = models.ForeignKey(Method, on_delete=models.CASCADE, null=True)
    reaction_id = models.ForeignKey(Reaction, on_delete=models.CASCADE, null=True)
    otsession_id = models.ForeignKey(OTSession, on_delete=models.CASCADE)
    wellindex = models.IntegerField()
    volume = models.FloatField(null=True)
    smiles = models.CharField(max_length=255, null=True)
    concentration = models.FloatField(null=True, blank=True)
    solvent = models.CharField(max_length=255, null=True)
    reactantfornextstep = models.BooleanField(default=False)
    available = models.BooleanField(default=True)


class CompoundOrder(models.Model):
    otsession_id = models.ForeignKey(
        OTSession, related_name="compoundorders", on_delete=models.CASCADE
    )
    ordercsv = models.FileField(upload_to="compoundorders/", max_length=255)


class OTScript(models.Model):
    otsession_id = models.ForeignKey(
        OTSession, related_name="otscripts", on_delete=models.CASCADE
    )
    otscript = models.FileField(upload_to="otscripts/", max_length=255)
