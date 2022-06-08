"""Create OT session"""
from __future__ import annotations
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.db.models import QuerySet

from statistics import median
from graphene_django import DjangoObjectType

from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
from pandas.core.frame import DataFrame


from car.models import (
    AnalyseAction,
    Batch,
    OTBatchProtocol,
    Reaction,
    Product,
    AddAction,
    OTSession,
    Deck,
    Plate,
    SolventPrep,
    Well,
    Pipette,
    TipRack,
    CompoundOrder,
)

import math
from .labwareavailable import labware_plates


class CreateOTSession(object):
    """
    Creates a StartOTSession object for generating a protocol
    from actions
    """

    def __init__(
        self,
        reactionstep: int,
        otbatchprotocolobj: OTBatchProtocol,
        sessiontype: str,
        reactiongroupqueryset: QuerySet[Reaction] = None,
    ):
        """Initiates a CreateOTSession

        Parameters
        ----------
        reactionstep: int
            The reaction step that the protocol is being created for
        otbatchprotocolobj: OTBatchProtocol
            The Django OTBactchProtocol model object, collects protocols
            for a group, that all OT sessions are linked to
        sessiontype: str
            The type of Ot Session being created eg. reaction, analyse
        reactiongroupqueryset: QuerySet[Reaction]
            The optional group of reactions that a reaction and analyse
            session needs to execute the actions for
        """
        self.reactionstep = reactionstep
        self.otbatchprotocolobj = otbatchprotocolobj
        self.otsessionqueryset = self.otbatchprotocolobj.otsessions.all()
        self.batchobj = Batch.objects.get(id=otbatchprotocolobj.batch_id_id)
        self.sessiontype = sessiontype
        if sessiontype == "reaction":
            self.createReactionSession(reactiongroupqueryset=reactiongroupqueryset)
        if sessiontype == "analyse":
            self.createAnalyseSession(reactiongroupqueryset=reactiongroupqueryset)

    def createAnalyseSession(self, reactiongroupqueryset: QuerySet[Reaction]):
        """Creates an analyse OT session

        Parameters
        ----------
        reactiongroupqueryset: QuerySet[Reaction]
            The group of reactions that an analyse
            session needs to execute the actions for
        """
        self.otsessionobj = self.createOTSessionModel()
        self.reactiongroupqueryset = reactiongroupqueryset
        self.reactionids = [
            reactionobj.id for reactionobj in self.reactiongroupqueryset
        ]
        self.allanalyseactionqueryset = self.getAnalyseActionQuerySet(
            reaction_ids=self.reactionids
        )
        self.productqueryset = self.getProductQuerySet(reaction_ids=self.reactionids)
        self.productsmiles = [productobj.smiles for productobj in self.productqueryset]

        self.analyseactionsdf = self.getAnalyseActionsDataFrame()
        self.roundedvolumes = self.getRoundedAnalyseActionVolumes(
            analyseactionqueryset=self.allanalyseactionqueryset
        )
        self.groupedanalysemethodobjs = self.getGroupedAnalyseMethods()
        self.deckobj = self.createDeckModel()
        self.numbertips = self.getNumberTips(queryset=self.allanalyseactionqueryset)
        self.tipracktype = self.getTipRackType(roundedvolumes=self.roundedvolumes)
        self.createTipRacks(tipracktype=self.tipracktype)
        self.pipettetype = self.getPipetteType(roundedvolumes=self.roundedvolumes)
        self.createPipetteModel()
        self.createSolventPlate(materialsdf=self.analyseactionsdf)
        previousreactionplates = self.getPreviousOTSessionReactionPlates()
        if previousreactionplates:
            self.cloneInputPlate(platesforcloning=previousreactionplates)
        for analysegroup in self.groupedanalysemethodobjs:
            # May need to rethink and do seprate sessions (Deck space) or optimise tip usage
            wellsneeded = len(analysegroup)
            method = analysegroup[0].method
            volumes = self.getRoundedAnalyseActionVolumes(analyseactionqueryset=analysegroup)
            platetype = self.getPlateType(platetype=method, volumes=volumes, wellsneeded=wellsneeded) 
            # self.platetype = self.getAnalysePlateType(
            #     wellsneeded=wellsneeded, method=method
            # )
            self.createAnalysePlate(
                method=method,
                platetype=platetype,
                analyseactionqueryset=analysegroup,
            )
         
    def createReactionSession(self, reactiongroupqueryset: QuerySet[Reaction]):
        """Creates a reaction OT session

        Parameters
        ----------
        reactiongroupqueryset: QuerySet[Reaction]
            The group of reactions that a reaction
            session needs to execute the actions for
        """
        self.otsessionobj = self.createOTSessionModel()
        self.reactiongroupqueryset = reactiongroupqueryset
        self.groupedtemperaturereactionobjs = self.getGroupedTemperatureReactions()
        self.alladdactionqueryset = [
            self.getAddActions(reaction_id=reactionobj.id)
            for reactionobj in self.reactiongroupqueryset
        ]
        self.alladdactionquerysetflat = [
            item for sublist in self.alladdactionqueryset for item in sublist
        ]
        self.roundedvolumes = self.getRoundedAddActionVolumes(
            addactionqueryset=self.alladdactionquerysetflat
        )
        self.deckobj = self.createDeckModel()
        self.numbertips = self.getNumberTips(queryset=self.alladdactionquerysetflat)
        self.tipracktype = self.getTipRackType(roundedvolumes=self.roundedvolumes)
        self.createTipRacks(tipracktype=self.tipracktype)
        self.pipettetype = self.getPipetteType(roundedvolumes=self.roundedvolumes)
        self.addactionsdf = self.getAddActionsDataFrame()
        inputplatequeryset = self.getInputPlatesNeeded()
        if inputplatequeryset:
            self.cloneInputPlate(platesforcloning=inputplatequeryset)
        self.createPipetteModel()
        self.createReactionStartingPlate()
        self.solventmaterialsdf = self.getAddActionsMaterialDataFrame(
            productexists=True
        )
        self.createSolventPlate(materialsdf=self.solventmaterialsdf)
        self.createReactionPlate()
        self.startingreactionplatequeryset = self.getStartingReactionPlateQuerySet(
            otsession_id=self.otsessionobj.id
        )

    def getAllPreviousOTSessionReactionPlates(self) -> QuerySet[Plate]:
        """Get all input reaction plates for all previous reaction OT sessions
        in a OT batch of protocols

        Returns
        -------
        allinputplatesqueryset: QuerySet[Plate]
            The plates used for all previous reaction OT
            sessions
        status: False
            The status if no plates were found
        """
        if self.otsessionqueryset:
            otsessionsids = [
                otsession.id
                for otsession in self.otsessionqueryset
                if otsession.sessiontype == "reaction"
            ]
            allinputplatesqueryset = Plate.objects.filter(
                otsession_id__in=otsessionsids, type="reaction"
            )
            return allinputplatesqueryset
        else:
            return False

    def getPlateQuerySet(self, otsession_id: int) -> QuerySet[Plate]:
        platequeryset = Plate.objects.filter(otsession_id=otsession_id)
        return platequeryset

    def getPreviousOTSessionReactionPlates(self) -> list[Plate]:
        """Checks if previous Reaction Plates exist and if they do
        check if the Wells match the products of the input
        reactiongroupqueryset for the session

        Returns
        -------
        previousreactionplates: list
            The list of previous OT session reaction plates in
            an OT batch protocol
        """
        previousotsessionobj = self.getPreviousObjEntry(
            queryset=self.otsessionqueryset, obj=self.otsessionobj
        )
        previousreactionplates = []
        if previousotsessionobj.sessiontype == "reaction":
            previousotsessionplates = self.getPlateQuerySet(
                otsession_id=previousotsessionobj.id
            )
            previousotsessioreactionplates = previousotsessionplates.filter(
                type="reaction"
            )
            for previousotsessionreactionplate in previousotsessioreactionplates:
                wellmatchqueryset = (
                    previousotsessionreactionplate.well_set.all()
                    .filter(
                        reaction_id__in=self.reactionids,
                        smiles__in=self.productsmiles,
                        type="reaction",
                    )
                    .distinct()
                )
                if wellmatchqueryset:
                    previousreactionplates.append(previousotsessionreactionplate)

        return previousreactionplates

    def getInputPlatesNeeded(self) -> list[Plate]:
        """Gets plates, created in previous reaction OT sessions,
        with reaction products that are required as reactants in
        current reaction session

        Returns
        -------
        inputplatesneeded: list
            The list of previous OT session reaction plates in
            an OT batch protocol that have products needed as
            reactants for current reaction OT session
        """
        inputplatesneeded = []
        allinputplatequerset = self.getAllPreviousOTSessionReactionPlates()
        methodids = [
            reactionobj.method_id for reactionobj in self.reactiongroupqueryset
        ]
        allreactantsmiles = [
            addaction.smiles for addaction in self.alladdactionquerysetflat
        ]
        if allinputplatequerset:
            for inputplateobj in allinputplatequerset:
                wellmatchqueryset = (
                    inputplateobj.well_set.all()
                    .filter(
                        method_id__in=methodids,
                        reactantfornextstep=True,
                        smiles__in=allreactantsmiles,
                        type="reaction",
                    )
                    .distinct()
                )
                if wellmatchqueryset:
                    inputplatesneeded.append(inputplateobj)
        return inputplatesneeded

    def getPreviousObjEntry(
        self, queryset: QuerySet, obj: DjangoObjectType
    ) -> DjangoObjectType:
        """Finds previous Django model object relative to the Django model
           object in a queryset

        Parameters
        ----------
        queryset: QuerySet
            The queryset to search for previous entries
        obj: DjangoObjectType
            The object that you want to find all previous object entries relative to

        Returns
        -------
        previousobj: DjangoObjectType
            The previous Django model object
        status: None
            None returned if no previous object is found
        """
        previousqueryset = queryset.filter(pk__lt=obj.pk).order_by("-pk")
        if previousqueryset:
            previousobj = previousqueryset[0]
            return previousobj
        else:
            return None

    def getPreviousObjEntries(
        self, queryset: QuerySet, obj: DjangoObjectType
    ) -> QuerySet:
        """Finds all previous Django model object relative to the Django model
           object in a queryset

        Parameters
        ----------
        queryset: QuerySet
            The queryset to search for previous entries
        obj: DjangoObjectType
            The object that you want to find all previous object entries relative to

        Returns
        -------
        previousqueryset: QuerySet
            The previous Django model objects as a queryset
        """
        previousqueryset = queryset.filter(pk__lt=obj.pk).order_by("-pk")
        return previousqueryset

    def getNextObjEntries(self, queryset: QuerySet, obj: DjangoObjectType) -> QuerySet:
        """Finds all proceeding Django model object relative to the Django model
           object in a queryset

        Parameters
        ----------
        queryset: QuerySet
            The queryset to search for proceeding entries
        obj: DjangoObjectType
            The object that you want to find all proceeding object entries relative to

        Returns
        -------
        nextqueryset: QuerySet
            The proceeding Django model objects as a queryset
        """
        nextqueryset = queryset.filter(pk__gt=obj.pk).order_by("pk")
        return nextqueryset

    def checkPreviousReactionProducts(self, reaction_id: int, smiles: str) -> bool:
        """Checks if any previous reactions had a product matching the smiles

        Parameters
        ----------
        reaction_id: int
            The reaction id of the Django model object to search for
            all relative previous reactions objects. The previosu reactions may
            have products that are this reaction's reactant input
        smiles: str
            The SMILES of the reaction's reactant and previous reaction products

        Returns
        -------
        status: bool
            The status is True if a match is found
        """
        reactionobj = self.getReaction(reaction_id=reaction_id)
        reactionqueryset = self.getReactionQuerySet(method_id=reactionobj.method_id.id)
        prevreactionqueryset = self.getPreviousObjEntries(
            queryset=reactionqueryset, obj=reactionobj
        )
        productmatches = []
        if prevreactionqueryset:
            for reactionobj in prevreactionqueryset:
                productobj = self.getProduct(reaction_id=reactionobj)
                if productobj.smiles == smiles:
                    productmatches.append(productobj)
            if productmatches:
                return True
            else:
                return False
        else:
            return False

    def checkNextReactionsAddActions(
        self, reactionobj: Reaction, productsmiles: str
    ) -> list:
        """Checks if there are any reaction objects following the reaction in a method.
           If there is, checks if any of the proceeding reaction add actions match
           the reaction product's SMILES

        Parameters
        ----------
        reactionobj: Reaction
            The Django reaction model object to search for it's product SMILES
            matching any add actions needing the product as a reactant in the
            proceeding reactions
        productsmiles: str
            The SMILES of the reaction's product

        Returns
        -------
        addactionsmatches: list
            The Django add action model objects that require the reaction product
            as an input reactant
        """
        reactionqueryset = self.getReactionQuerySet(method_id=reactionobj.method_id.id)
        nextreactionqueryset = self.getNextObjEntries(
            queryset=reactionqueryset, obj=reactionobj
        )
        addactionsmatches = []
        for reactionobj in nextreactionqueryset:
            addactionmatch = self.getAddActions(reaction_id=reactionobj.id).filter(
                materialsmiles=productsmiles
            )
            if addactionmatch:
                addactionsmatches.append(addactionmatch[0])
        return addactionsmatches

    def getAnalyseActionQuerySet(self, reaction_ids: list) -> QuerySet[AnalyseAction]:
        """Get the analyse action queryset linked to a list reaction ids

        Parameters
        ----------
        reaction_ids: list
            The list of reaction ids to search for matching analyse action
            objects

        Returns
        -------
        analyseactionqueryset: QuerySet[AnalyseAction]
            The Django queyset of the analyse actions for the reaction ids
        """
        analyseactionqueryset = AnalyseAction.objects.filter(
            reaction_id__in=reaction_ids,
        )
        return analyseactionqueryset

    def getReaction(self, reaction_id: int) -> Reaction:
        """Get reaction object

        Parameters
        ----------
        reaction_id: int
            The reaction id to search for a reaction

        Returns
        -------
        reactionobj: Reaction
            The reaction Django model object
        """
        reactionobj = Reaction.objects.get(id=reaction_id)
        return reactionobj

    def getReactionQuerySet(self, method_id: int) -> QuerySet[Reaction]:
        """Get reaction queryset for method_id

        Parameters
        ----------
        method_id: int
            The method id to search for related reactions

        Returns
        -------
        reactionqueryset: QuerySet[Reaction]
            The reaction queryset related to the method id
        """
        reactionqueryset = Reaction.objects.filter(method_id=method_id)
        return reactionqueryset

    def getProductQuerySet(self, reaction_ids: list) -> QuerySet[Product]:
        """Get product queryset for reaction ids

        Parameters
        ----------
        reaction_ids: list
            The reaction ids to search for related products

        Returns
        -------
        productqueryset: QuerySet[Product]
            The product queryset related to the reaction ids
        """
        productqueryset = Product.objects.filter(reaction_id__in=reaction_ids)
        return productqueryset

    def getProduct(self, reaction_id: int) -> Product:
        """Get product object

        Parameters
        ----------
        reaction_id: int
            The reaction id to search for a matching product

        Returns
        -------
        productobj: Product
            The product Django model object
        """
        productobj = Product.objects.get(reaction_id=reaction_id)
        return productobj

    def getAddActions(self, reaction_id: int) -> QuerySet[AddAction]:
        """Get add actions queryset for reaction_id

        Parameters
        ----------
        reaction_id: int
            The reaction to search for related add actions

        Returns
        -------
        addactionqueryset: QuerySet[AddAction]
            The add actions related to the reaction
        """
        addactionqueryset = AddAction.objects.filter(reaction_id=reaction_id).order_by(
            "id"
        )
        return addactionqueryset

    def getStartingReactionPlateQuerySet(self, otsession_id: int) -> QuerySet[Plate]:
        """Get all plates used in reactions for an OT session

        Parameters
        ----------
        otsession_id: int
            The OT session to search for all the reaction plates

        Returns
        -------
        reactionplatequeryset: QuerySet[Plate]
            The plates used in reactions in an OT session
        """
        reactionplatequeryset = Plate.objects.filter(
            otsession_id=otsession_id, name__contains="Reactionplate"
        ).order_by("id")
        return reactionplatequeryset

    def getRoundedAnalyseActionVolumes(
        self, analyseactionqueryset: QuerySet[AnalyseAction]
    ) -> list:
        """Gets the total rounded volume (ul), for sample and solvent volume used,
           for all the analyse actions

        Parameters
        ----------
        analyseactionqueryset: QuerySet[AnalyseAction]
            The analyse actions to calculate the rounded volumes (ul)

        Returns
        -------
        roundedvolumes: list
            The list of rounded volumes for the analyse actions
        """
        roundedvolumes = [
            round(analyseactionobj.samplevolume + analyseactionobj.solventvolume)
            for analyseactionobj in analyseactionqueryset
        ]
        return roundedvolumes

    def getRoundedAddActionVolumes(
        self, addactionqueryset: QuerySet[AddAction]
    ) -> list:
        """Gets the total rounded volume (ul) for all the add actions

        Parameters
        ----------
        addactionqueryset: QuerySet[AddAction]
            The add actions to cacluate the rounded volumes (ul)

        Returns
        -------
        roundedvolumes: list
            The list of rounded volumes for the add actions
        """
        roundedvolumes = [
            round(addactionobj.volume) for addactionobj in addactionqueryset
        ]
        return roundedvolumes

    def getTipRackType(self, roundedvolumes: list) -> str:
        """Gets OT tiprack best suited for transferring volumes (ul)
           that minimises the number of tranfers required

        Parameters
        ----------
        roundedvolumes: list
            The list of rounded volumes needed for a set of actions

        Returns
        -------
        tipracktype: str
            The most suitable tiprack type
        """
        tipsavailable = {
            300: "opentrons_96_tiprack_300ul",
            10: "opentrons_96_tiprack_20ul",
        }
        tipkey = min(
            tipsavailable,
            key=lambda x: self.getNumberTransfers(
                pipettevolume=x, roundedvolumes=roundedvolumes
            ),
        )
        tipracktype = tipsavailable[tipkey]
        return tipracktype

    def getPipetteType(self, roundedvolumes: list) -> str:
        """Gets the type of pippete that minmises the number of tranfers
           needed for transferring volumes (ul).

        Parameters
        ----------
        roundedvolumes: list
            The list of rounded volumes that need to be transferred

        Returns
        -------
        pipettetype: str
            The pipette type needed
        """
        pipettesavailable = {
            10: {
                "labware": "p10_single",
                "position": "right",
                "type": "single",
                "maxvolume": 10,
            },
            300: {
                "labware": "p300_single",
                "position": "right",
                "type": "single",
                "maxvolume": 300,
            },
        }
        pipettekey = min(
            pipettesavailable,
            key=lambda x: self.getNumberTransfers(
                pipettevolume=x, roundedvolumes=roundedvolumes
            ),
        )
        pipettetype = pipettesavailable[pipettekey]
        return pipettetype

    def getNumberTips(self, queryset: QuerySet) -> int:
        """Gets the number of tips required for transferring
           actions

        Parameters
        ----------
        queryset: Queryset
            The set of actions that need to be transferred

        Returns
        -------
        numbertips: int
            The number of tips required
        """
        numbertips = len(queryset)
        return numbertips

    def getNumberTransfers(self, pipettevolume: int, roundedvolumes: list) -> int:
        """Gets the number of transfers required for transferring
           a list of rounded volumes

        Parameters
        ----------
        pipettevolume: int
            The pippette's maximum transfer volume (ul)
        roundedvolumes: list
            The list fo rounded volumes that need to be transferred

        Returns
        -------
        numbertransfers: int
            The number of tranfers required for the pipette type used
        """
        numbertransfers = sum(
            [
                round(volume / pipettevolume) if pipettevolume < volume else 1
                for volume in roundedvolumes
            ]
        )
        return numbertransfers

    def getAnalyseActionsDataFrame(self) -> DataFrame:
        """Creates a Pandas dataframe from the analyse actions.
           Current version of code uses a dataframe to create plates and wells.

        Returns
        -------
        analyseactionsdf: DataFrame
            The dataframe of the analyse actions
        """
        # Optimise -> https://stackoverflow.com/questions/11697887/converting-django-queryset-to-pandas-dataframe
        analyseactionsdf = pd.DataFrame(list(self.allanalyseactionqueryset.values()))
        analyseactionsdf = analyseactionsdf.rename(columns={"solventvolume": "volume"})
        return analyseactionsdf

    def getAddActionsDataFrame(self) -> DataFrame:
        """Creates a Pandas dataframe from the add actions.
           Current version of code uses a dataframe to create plates and wells.

        Returns
        -------
        addactionsdf: DataFrame
            The dataframe of the add actions
        """
        # Optimise -> https://stackoverflow.com/questions/11697887/converting-django-queryset-to-pandas-dataframe
        addactionslistdf = []

        for addactionqueryset in self.alladdactionqueryset:
            addactionsdf = pd.DataFrame(list(addactionqueryset.values()))
            if not addactionsdf.empty:
                addactionslistdf.append(addactionsdf)
        addactionsdf = pd.concat(addactionslistdf)

        addactionsdf["uniquesolution"] = addactionsdf.apply(
            lambda row: self.combinestrings(row), axis=1
        )

        return addactionsdf

    def getMaxWellVolume(self, plateobj: Plate) -> float:
        """Get max well volume of a well plate

        Parameters
        ----------
        plateobj: Plate
            The plate to get the max well volume of

        Returns
        -------
        maxwellvolume: float
            The maximum well volume of a well plate
        """
        maxwellvolume = plateobj.maxwellvolume
        return maxwellvolume

    def getDeadVolume(self, maxwellvolume: float) -> float:
        """Calculates the dead volume (5%) of a well

        Parameters
        ----------
        maxwellvolume: float
            The well's maximum volume

        Returns
        -------
        deadvolume: float
            The dead volume of the well
        """
        deadvolume = maxwellvolume * 0.05
        return deadvolume

    def getCloneWells(self, plateobj: Plate) -> QuerySet[Well]:
        """Retrieves the wells for a plate

        Parameters
        ----------
        plateobj: Plate
            The plate to get all the related wells to

        Returns
        -------
        clonewellqueryset: QuerySet[Well]
            The plates wells
        """
        clonewellqueryset = Well.objects.filter(plate_id=plateobj.id)
        return clonewellqueryset

    def getUniqueAnalyseMethods(self) -> list:
        """Get the methods related to analyse actions

        Returns
        -------
        methods: list
            The unique set of methods related to the analyse
            actions
        """

        methods = sorted(
            set(
                [
                    analyseactionobj.method
                    for analyseactionobj in self.allanalyseactionqueryset
                ]
            )
        )
        return methods

    def getUniqueTemperatures(self) -> list:
        """Set of reaction temperatures

        Returns
        ------
        temperatures: list
            The set of reaction temperatures
        """
        temperatures = sorted(
            set([reactionobj.temperature for reactionobj in self.reactiongroupqueryset])
        )
        return temperatures

    def getGroupedAnalyseMethods(self) -> list:
        """Group analyse actions by method

        Returns
        -------
        groupedanalysemethodobjs: list
            List of sublists of analyse actions grouped by synthesis method
        """
        methods = self.getUniqueAnalyseMethods()
        groupedanalysemethodobjs = []

        for method in methods:
            analysemethodgroup = [
                analyseactionobj
                for analyseactionobj in self.allanalyseactionqueryset
                if analyseactionobj.method == method
            ]
            groupedanalysemethodobjs.append(analysemethodgroup)

        return groupedanalysemethodobjs

    def getGroupedTemperatureReactions(self) -> list:
        """Group reactions done at the same temperature

        Returns
        -------
        groupedtemperaturereactionobjs: list
            The list of sublists of reactions grouped by reaction
            temperature
        """
        temperatures = self.getUniqueTemperatures()
        groupedtemperaturereactionobjs = []

        for temperature in temperatures:
            temperaturereactiongroup = [
                reactionobj
                for reactionobj in self.reactiongroupqueryset
                if reactionobj.temperature == temperature
            ]
            groupedtemperaturereactionobjs.append(temperaturereactiongroup)

        return groupedtemperaturereactionobjs

    def getMedianValue(self, values: list) -> float:
        medianvalue = median(values)
        return medianvalue

    def getMaxValue(self, values: list) -> float:
        maxvalue = max(values)
        return maxvalue

    def getSumValue(self, values: list) -> float:
        sumvalue = sum(values)
        return sumvalue

    def getNumberObjs(self, queryset: list) -> int:
        numberobjs = len(queryset)
        return numberobjs

    def getReactionVolumes(self, grouptemperaturereactionobjs: list) -> list:
        reactionvolumes = []
        for reactionobj in grouptemperaturereactionobjs:
            addactionqueryset = self.getAddActions(reaction_id=reactionobj.id)
            roundedvolumes = self.getRoundedAddActionVolumes(
                addactionqueryset=addactionqueryset
            )
            sumvolume = self.getSumValue(values=roundedvolumes)
            reactionvolumes.append(sumvolume)
        return reactionvolumes

    def getReactionPlateType(self, grouptemperaturereactionobjs: list) -> str:
        """Gets the labware plate type needed based on a grouped list of
           reactions needing the same temperature. Plate selected based on getting
           closest well volume to the median volume of the

        XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

        Parameters
        ----------
        grouptemperaturereactionobjs: list
            The list of grouped reactions done at the same temperature

        Returns
        -------
        labwareplatetype: str
            The type of labware plate required
        """
        reactionvolumes = []
        for reactionobj in grouptemperaturereactionobjs:
            addactionqueryset = self.getAddActions(reaction_id=reactionobj.id)
            roundedvolumes = self.getRoundedAddActionVolumes(
                addactionqueryset=addactionqueryset
            )
            sumvolume = self.getSumValue(values=roundedvolumes)
            reactionvolumes.append(sumvolume)
        maxvolume = self.getMaxValue(values=reactionvolumes)
        medianvolume = self.getMedianValue(values=reactionvolumes)
        headspacevolume = maxvolume + (maxvolume * 0.2)
        reactiontemperature = grouptemperaturereactionobjs[0].temperature

        if reactiontemperature > 25:
            labwareplatetypes = [
                labware_plate
                for labware_plate in labware_plates
                if labware_plates[labware_plate]["reflux"]
                and labware_plates[labware_plate]["volume_well"] > headspacevolume
            ]
        else:
            labwareplatetypes = [
                labware_plate
                for labware_plate in labware_plates
                if not labware_plates[labware_plate]["reflux"]
                and labware_plates[labware_plate]["volume_well"] > headspacevolume
            ]

        if len(labwareplatetypes) > 1:
            volumewells = [
                labware_plates[labware_plate]["volume_well"]
                for labware_plate in labwareplatetypes
            ]
            indexclosestvalue = min(
                range(len(volumewells)), key=lambda x: abs(x - medianvolume)
            )
            labwareplatetype = labwareplatetypes[indexclosestvalue]
        else:
            labwareplatetype = labwareplatetypes[0]

        return labwareplatetype

    def getNumberVials(self, maxvolumevial: float, volumematerial: float) -> int:
        """Gets the total number of vials needed to prepare a starter plate

        Parameters
        ----------
        maxvolumevial: float
            The maximum volume of a vial
        volumematerial: float
            The volume of the material that needs to be stored in a vial

        Returns
        -------
        novialneeded: int
            The number of vials needed to store the material
        """
        if maxvolumevial > volumematerial:
            novialsneeded = 1
        else:
            volumestoadd = []
            deadvolume = self.getDeadVolume(maxwellvolume=maxvolumevial)
            novialsneededratio = volumematerial / (maxvolumevial - deadvolume)
            frac, whole = math.modf(novialsneededratio)
            volumestoadd = [maxvolumevial for maxvolumevial in range(int(whole))]
            volumestoadd.append(frac * maxvolumevial + deadvolume)
            novialsneeded = sum(volumestoadd)
        return novialsneeded

    def getPlateType(self, platetype: str, volumes: list, temperature: int = 25, wellsneeded: int = None):
        """Gets best suited plate
           Optimises for (in order of decreasing preference) minimum: 
           number of plates, volume difference and mumber of vials 
        """
        platetype = platetype.lower()
        maxvolume = self.getMaxValue(values=volumes)
        medianvolume = self.getMedianValue(values=volumes)
        headspacevolume = maxvolume + (maxvolume * 0.2)
        
        possiblelabwareplatetypes = [
            labware_plate
            for labware_plate in labware_plates
            if platetype in labware_plates[labware_plate]["type"] 
            and labware_plates[labware_plate]["max_temp"] >= temperature
            and labware_plates[labware_plate]["volume_well"] >= headspacevolume
        ]

        vialcomparedict = {}

        for labwareplate in possiblelabwareplatetypes:
            maxtemp = labware_plates[labwareplate]["max_temp"]
            maxvolumevial = labware_plates[labwareplate]["volume_well"]
            noplatevials = labware_plates[labwareplate]["no_wells"]
            if not wellsneeded:
                wellsneeded = sum([self.getNumberVials(
                        maxvolumevial=maxvolumevial, volumematerial=volume) for volume in volumes])
            noplatesneeded = int(math.ceil(wellsneeded / noplatevials))
            volumedifference = maxvolumevial - medianvolume
            tempdifference = maxtemp - temperature
            vialcomparedict[labwareplate] = {"noplatesneeded": noplatesneeded, "volumedifference": volumedifference, "novialsneeded": wellsneeded, "tempdifference": tempdifference} 

        mininumnoplatesneeeded = min((d["noplatesneeded"] for d in vialcomparedict.values()))
        mininumtempdifference = min((d["tempdifference"] for d in vialcomparedict.values()))

        labwareplatetypes = [
            labwareplate 
            for labwareplate in vialcomparedict 
            if vialcomparedict[labwareplate]["noplatesneeded"]==mininumnoplatesneeeded 
            and vialcomparedict[labwareplate]["tempdifference"]==mininumtempdifference
        ]
        if len(labwareplatetypes) > 1:
            mininumvolumedifference = min((d["volumedifference"] for d in vialcomparedict.values()))
            labwareplatetypes =  [labwareplate for labwareplate in vialcomparedict if vialcomparedict[labwareplate]["volumedifference"]==mininumvolumedifference]
            if len(labwareplatetypes) > 1:
                mininumnovialsneeded = min((d["novialsneeded"] for d in vialcomparedict.values()))
                labwareplatetypes =  [labwareplate for labwareplate in vialcomparedict if vialcomparedict[labwareplate]["novialsneeded"]==mininumnovialsneeded]
        
        return labwareplatetypes[0]

    def getAnalysePlateType(self, wellsneeded: int, method: str):
        """Finds best labware plate for analysis"""
        analyselc = method.lower()

        labwareplatetypes = [
            labware_plate
            for labware_plate in labware_plates
            if analyselc in labware_plates[labware_plate]["type"]
        ]

        wellcomparedict = {}

        for labwareplate in labwareplatetypes:
            maxvolumevial = labware_plates[labwareplate]["volume_well"]
            noplatevials = labware_plates[labwareplate]["no_wells"]
            platesneeded = int(math.ceil(wellsneeded / noplatevials))
            wellcomparedict[maxvolumevial] = platesneeded
            minimumnovialsvolume = min(wellcomparedict, key=wellcomparedict.get)

        labwareplatetype = [
            labware_plate
            for labware_plate in labwareplatetypes
            if labware_plates[labware_plate]["volume_well"] == minimumnovialsvolume
        ][0]

        return labwareplatetype

    def getStartingMaterialPlateType(self, platetype: str, materialsdf: DataFrame):
        """Gets best suited plate for a list of materials and volumes
           Optimises for least amount of plates, wells and closest volume well 
           vs volume material needed
        """
        labwareplatetypes = [
            labware_plate
            for labware_plate in labware_plates
            if platetype in labware_plates[labware_plate]["type"]
        ]

        vialcomparedict = {}

        medianvolume = self.getMedianValue(values=materialsdf["volume"])

        for labwareplate in labwareplatetypes:
            maxvolumevial = labware_plates[labwareplate]["volume_well"]
            noplatevials = labware_plates[labwareplate]["no_wells"]

            vialsneeded = materialsdf.apply(
                lambda row: self.getNumberVials(
                    maxvolumevial=maxvolumevial, volumematerial=row["volume"]
                ),
                axis=1,
            )
            totalvialsneeded = sum(vialsneeded)
            noplatesneeded = int(math.ceil(totalvialsneeded / noplatevials))
            volumedifference = maxvolumevial - medianvolume
            vialcomparedict[labwareplate] = {"noplatesneeded": noplatesneeded, "volumedifference": volumedifference, "novialsneeded": totalvialsneeded} 

        mininumnoplatesneeeded = min((d["noplatesneeded"] for d in vialcomparedict.values()))
        labwareplatetypes = [labwareplate for labwareplate in vialcomparedict if vialcomparedict[labwareplate]["noplatesneeded"]==mininumnoplatesneeeded]
        if len(labwareplatetypes) > 1:
            mininumvolumedifference = min((d["volumedifference"] for d in vialcomparedict.values()))
            labwareplatetypes =  [labwareplate for labwareplate in vialcomparedict if vialcomparedict[labwareplate]["volumedifference"]==mininumvolumedifference]
            if len(labwareplatetypes) > 1:
                mininumnovialsneeded = min((d["novialsneeded"] for d in vialcomparedict.values()))
                labwareplatetypes =  [labwareplate for labwareplate in vialcomparedict if vialcomparedict[labwareplate]["novialsneeded"]==mininumnovialsneeded]
        
        return labwareplatetypes[0]

    def getAddActionsMaterialDataFrame(self, productexists: bool) -> DataFrame:
        """Aggegates all adda actions materials and sums up volume requires using solvent type and
        concentration

        Parameters
        ----------
        productexists: bool
            Set to true to check if the add action material needed is a product from
            a previous reaction

        Returns
        -------
        materialsdf: DataFrame
            The add action material as dataframe grouping materials by SMILES, concentration
            and solvent
        """
        materialsdf = self.addactionsdf.groupby(["uniquesolution"]).agg(
            {
                "reaction_id_id": "first",
                "smiles": "first",
                "volume": "sum",
                "solvent": "first",
                "concentration": "first",
            }
        )
        materialsdf["productexists"] = materialsdf.apply(
            lambda row: self.checkPreviousReactionProducts(
                reaction_id=row["reaction_id_id"], smiles=row["smiles"]
            ),
            axis=1,
        )

        if productexists:
            materialsdf = materialsdf[materialsdf["productexists"]]

        if not productexists:
            materialsdf = materialsdf[~materialsdf["productexists"]]

        materialsdf = materialsdf.sort_values(["solvent", "volume"], ascending=False)
        return materialsdf

    def createOTSessionModel(self):
        """Create an OT Session object"""
        otsessionobj = OTSession()
        otsessionobj.otbatchprotocol_id = self.otbatchprotocolobj
        otsessionobj.reactionstep = self.reactionstep
        otsessionobj.sessiontype = self.sessiontype
        otsessionobj.save()
        return otsessionobj

    def createDeckModel(self):
        """Create a deck object"""
        deckobj = Deck()
        deckobj.otsession_id = self.otsessionobj
        deckobj.numberslots = 11
        deckobj.save()
        self.deckobj = deckobj
        return deckobj

    def createPipetteModel(self):
        """Cretae a pipette object"""
        pipetteobj = Pipette()
        pipetteobj.otsession_id = self.otsessionobj
        pipetteobj.position = self.pipettetype["position"]
        pipetteobj.maxvolume = self.pipettetype["maxvolume"]
        pipetteobj.type = self.pipettetype["type"]
        pipetteobj.name = "{}_{}".format(
            self.pipettetype["position"], self.pipettetype["labware"]
        )
        pipetteobj.labware = self.pipettetype["labware"]
        pipetteobj.save()

    def createTiprackModel(self, name: str):
        """Creates TipRack object"""
        indexslot = self.checkDeckSlotAvailable()
        if indexslot:
            index = indexslot
            tiprackobj = TipRack()
            tiprackobj.otsession_id = self.otsessionobj
            tiprackobj.deck_id = self.deckobj
            tiprackobj.name = "{}_{}".format(name, indexslot)
            tiprackobj.index = index
            tiprackobj.labware = name
            tiprackobj.save()
        else:
            print("No more deck slots available")

    def createPlateModel(self, platetype: str, platename: str, labwaretype: str):
        """Creates Plate model if Deck index is available"""
        indexslot = self.checkDeckSlotAvailable()
        if indexslot:
            plateindex = indexslot
            maxwellvolume = labware_plates[labwaretype]["volume_well"]
            numberwells = labware_plates[labwaretype]["no_wells"]
            plateobj = Plate()
            plateobj.otsession_id = self.otsessionobj
            plateobj.deck_id = self.deckobj
            plateobj.labware = labwaretype
            plateobj.index = plateindex
            plateobj.name = "Reaction_step_{}_{}_index_{}".format(
                self.reactionstep, platename, indexslot
            )
            plateobj.type = platetype
            plateobj.maxwellvolume = maxwellvolume
            plateobj.numberwells = numberwells
            plateobj.save()
            return plateobj
        else:
            print("No more deck slots available")

    def createWellModel(
        self,
        plateobj: Plate,
        welltype: str,
        wellindex: int,
        volume: float = None,
        reactionobj: Reaction = None,
        smiles: str = None,
        concentration: float = None,
        solvent: str = None,
        reactantfornextstep: bool = False,
    ) -> Well:
        """Creates a well object

        Parameters
        ----------
        plateobj: Plate
            The plate that the well is linked to
        welltype: str
            The well type eg. reaction, analyse
        wellindex: int
            The index of the well in the plate
        volume: float = None
            The optional volume of the well contents
        reactionobj: Reaction = None
            The optional reaction the well is linked to
        smiles: str = None
            The optional contents of the well
        concentration: float = None
            The optional cocentration of the well contents
        solvent: str = None
            The optional solvent used to prepare the content of the well
        reactantfornextstep: bool = False
            The optional setting if the contents of the well are
            used in any proceeding reactions

        Returns
        -------
        wellobj: Well
            The well created
        """
        wellobj = Well()
        wellobj.otsession_id = self.otsessionobj
        wellobj.plate_id = plateobj
        if reactionobj:
            wellobj.reaction_id = reactionobj
            wellobj.method_id = reactionobj.method_id
        wellobj.type = welltype
        wellobj.index = wellindex
        wellobj.volume = volume
        wellobj.smiles = smiles
        wellobj.concentration = concentration
        wellobj.solvent = solvent
        wellobj.reactantfornextstep = reactantfornextstep
        wellobj.save()
        return wellobj

    def calcMass(self, row) -> float:
        """Calculates the mass of material (mg) from the
           concentration (mol/L) and volume (ul) needed

        Parameters
        ----------
        row: DataFrame row
            The row from the dataframe containing the
            concentration and volume information

        Retruns
        -------
        massmg: float
            The mass of the material needed
        """
        mols = row["concentration"] * row["amount-ul"] * 1e-6
        smiles = row["SMILES"]
        mw = Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles))
        massmg = mols * mw * 1e3
        return round(massmg, 2)

    def createCompoundOrderModel(self, orderdf: DataFrame):
        """Creates a compound order object"""
        compoundorderobj = CompoundOrder()
        compoundorderobj.otsession_id = self.otsessionobj
        csvdata = orderdf.to_csv(encoding="utf-8", index=False)
        ordercsv = default_storage.save(
            "compoundorders/"
            + "{}-session-starterplate-for-batch-{}-reactionstep-{}-sessionid-{}".format(
                self.sessiontype,
                self.batchobj.batchtag,
                self.reactionstep,
                str(self.otsessionobj.id),
            )
            + ".csv",
            ContentFile(csvdata),
        )
        compoundorderobj.ordercsv = ordercsv
        compoundorderobj.save()

    def createSolventPrepModel(self, solventdf: DataFrame):
        """Creates a Django solvent prep object - a solvent prep file

        Parameters
        ----------
        solventdf: DataFrame
            The solvent dataframe grouped by type of solvent
            and contains the: platenames, well index, solvent
            and volume required
        """
        solventprepobj = SolventPrep()
        solventprepobj.otsession_id = self.otsessionobj
        csvdata = solventdf.to_csv(encoding="utf-8", index=False)
        ordercsv = default_storage.save(
            "solventprep/"
            + "{}-session-solventplate-for-batch-{}-reactionstep-{}-sessionid-{}".format(
                self.sessiontype,
                self.batchobj.batchtag,
                self.reactionstep,
                str(self.otsessionobj.id),
            )
            + ".csv",
            ContentFile(csvdata),
        )
        solventprepobj.solventprepcsv = ordercsv
        solventprepobj.save()

    def createTipRacks(self, tipracktype: str):
        """Create the tipracks needed

        Parameters
        ----------
        tipracktype: str
            The type of tiprack needed
        """
        numberacks = int(-(-self.numbertips // 96))
        for rack in range(numberacks):
            self.createTiprackModel(name=tipracktype)

    def checkDeckSlotAvailable(self) -> int:
        """Check if a deck slot is available

        Returns
        -------
        testslotavailable: int
            The index of the deck slot available
        status: False
            Returns false if no deck slot/index is available
        """
        testslotavailable = self.deckobj.indexslotavailable
        if testslotavailable <= self.deckobj.numberslots:
            self.deckobj.indexslotavailable = testslotavailable + 1
            self.deckobj.save()
            return testslotavailable
        else:
            self.deckobj.slotavailable = False
            self.deckobj.save()
            return False

    def checkPlateWellsAvailable(self, plateobj: Plate) -> int:
        """Check if any wells available on a plate

        Parameters
        ----------
        plateobj: Plate
            The plate to search for a well available

        Returns
        -------
        plateobj.indexswellavailable: int
            The index of the well available on a plate
        status: False
            Returns false if no well is available
        """
        wellavailable = plateobj.indexswellavailable + 1
        numberwells = plateobj.numberwells
        if wellavailable <= numberwells:
            plateobj.indexswellavailable = wellavailable
            plateobj.save()
            return plateobj.indexswellavailable
        else:
            plateobj.wellavailable = False
            plateobj.save()
            return False


    def createAnalysePlate(
        self, method: str, platetype: str, analyseactionqueryset: list
    ):
        """Creates analyse plate/s for executing analyse actions"""
        plateobj = self.createPlateModel(
            platetype=platetype.lower(),
            platename="{}_analyse_plate".format(method),
            labwaretype=platetype,
        )

        for analyseactionobj in analyseactionqueryset:
            reactionobj = analyseactionobj.reaction_id
            productobj = self.getProduct(reaction_id=reactionobj.id)
            volume = analyseactionobj.samplevolume + analyseactionobj.solventvolume
            indexwellavailable = self.checkPlateWellsAvailable(plateobj=plateobj)
            if not indexwellavailable:
                plateobj = self.createPlateModel(
                    platename="{}_analyse_plate".format(method),
                    labwaretype=platetype,
                )

                indexwellavailable = self.checkPlateWellsAvailable(plateobj=plateobj)

            self.createWellModel(
                plateobj=plateobj,
                reactionobj=reactionobj,
                welltype="analyse",
                wellindex=indexwellavailable - 1,
                volume=volume,
                smiles=productobj.smiles,
            )

    def createReactionStartingPlate(self):
        """Creates the starting material plate/s for executing a reaction's add actions"""
        startingmaterialsdf = self.getAddActionsMaterialDataFrame(productexists=False)

        startinglabwareplatetype = self.getPlateType(platetype="startingmaterial", volumes=startingmaterialsdf["volume"])
        # startinglabwareplatetype = self.getStartingMaterialPlateType(
        #     platetype="startingmaterial",
        #     materialsdf=startingmaterialsdf
        # )

        plateobj = self.createPlateModel(
            platetype="startingmaterial",
            platename="Startingplate",
            labwaretype=startinglabwareplatetype,
        )

        maxwellvolume = self.getMaxWellVolume(plateobj=plateobj)
        deadvolume = self.getDeadVolume(maxwellvolume=maxwellvolume)

        orderdictslist = []

        for i in startingmaterialsdf.index.values:
            extraerrorvolume = startingmaterialsdf.at[i, "volume"] * 0.1
            totalvolume = startingmaterialsdf.at[i, "volume"] + extraerrorvolume
            if totalvolume > maxwellvolume:
                nowellsneededratio = totalvolume / (maxwellvolume - deadvolume)

                frac, whole = math.modf(nowellsneededratio)
                volumestoadd = [maxwellvolume for i in range(int(whole))]
                volumestoadd.append(frac * maxwellvolume + deadvolume)

                for volumetoadd in volumestoadd:
                    indexwellavailable = self.checkPlateWellsAvailable(
                        plateobj=plateobj
                    )
                    if not indexwellavailable:

                        plateobj = self.createPlateModel(
                            platetype="startingmaterial",
                            platename="Startingplate",
                            labwaretype=startinglabwareplatetype,
                        )

                        indexwellavailable = self.checkPlateWellsAvailable(
                            plateobj=plateobj
                        )

                    wellobj = self.createWellModel(
                        plateobj=plateobj,
                        reactionobj=self.getReaction(
                            reaction_id=startingmaterialsdf.at[i, "reaction_id_id"]
                        ),
                        welltype="startingmaterial",
                        wellindex=indexwellavailable - 1,
                        volume=volumetoadd,
                        smiles=startingmaterialsdf.at[i, "smiles"],
                        concentration=startingmaterialsdf.at[i, "concentration"],
                        solvent=startingmaterialsdf.at[i, "solvent"],
                    )

                    orderdictslist.append(
                        {
                            "SMILES": startingmaterialsdf.at[i, "smiles"],
                            "name": plateobj.name,
                            "labware": plateobj.labware,
                            "well": wellobj.index,
                            "concentration": startingmaterialsdf.at[i, "concentration"],
                            "solvent": startingmaterialsdf.at[i, "solvent"],
                            "amount-ul": round(volumetoadd, 2),
                        }
                    )

            else:
                indexwellavailable = self.checkPlateWellsAvailable(plateobj=plateobj)
                volumetoadd = totalvolume + deadvolume

                if not indexwellavailable:
                    plateobj = self.createPlateModel(
                        platetype="startingmaterial",
                        platename="Startingplate",
                        labwaretype=startinglabwareplatetype,
                    )
                    indexwellavailable = self.checkPlateWellsAvailable(
                        plateobj=plateobj
                    )

                wellobj = self.createWellModel(
                    plateobj=plateobj,
                    reactionobj=self.getReaction(
                        reaction_id=startingmaterialsdf.at[i, "reaction_id_id"]
                    ),
                    welltype="startingmaterial",
                    wellindex=indexwellavailable - 1,
                    volume=volumetoadd,
                    smiles=startingmaterialsdf.at[i, "smiles"],
                    concentration=startingmaterialsdf.at[i, "concentration"],
                    solvent=startingmaterialsdf.at[i, "solvent"],
                )

                orderdictslist.append(
                    {
                        "SMILES": startingmaterialsdf.at[i, "smiles"],
                        "name": plateobj.name,
                        "labware": plateobj.labware,
                        "well": wellobj.index,
                        "concentration": startingmaterialsdf.at[i, "concentration"],
                        "solvent": startingmaterialsdf.at[i, "solvent"],
                        "amount-ul": round(volumetoadd, 2),
                    }
                )

        orderdf = pd.DataFrame(orderdictslist)
        orderdf["mass-mg"] = orderdf.apply(lambda row: self.calcMass(row), axis=1)

        self.createCompoundOrderModel(orderdf=orderdf)

    def createReactionPlate(self):
        """Creates reaction plate/s for executing reaction's add actions"""
        for grouptemperaturereactionobjs in self.groupedtemperaturereactionobjs:
            reactiontemperature = grouptemperaturereactionobjs[0].temperature
            volumes  = self.getReactionVolumes(grouptemperaturereactionobjs=grouptemperaturereactionobjs)
            labwareplatetype = self.getPlateType(platetype="reaction", temperature=reactiontemperature, volumes=volumes)
            # labwareplatetype = self.getReactionPlateType(
            #     grouptemperaturereactionobjs=grouptemperaturereactionobjs
            # )

            plateobj = self.createPlateModel(
                platetype="reaction",
                platename="Reactionplate",
                labwaretype=labwareplatetype,
            )

            for reactionobj in grouptemperaturereactionobjs:
                productobj = self.getProduct(reaction_id=reactionobj.id)
                indexwellavailable = self.checkPlateWellsAvailable(plateobj=plateobj)
                if not indexwellavailable:
                    plateobj = self.createPlateModel(
                        platetype="reaction",
                        platename="Reactionplate",
                        labwaretype=labwareplatetype,
                    )

                    indexwellavailable = self.checkPlateWellsAvailable(
                        plateobj=plateobj
                    )

                nextaddactionobjs = self.checkNextReactionsAddActions(
                    reactionobj=reactionobj, productsmiles=productobj.smiles
                )
                if nextaddactionobjs:
                    nextaddactionobj = nextaddactionobjs[0]
                    self.createWellModel(
                        plateobj=plateobj,
                        reactionobj=reactionobj,
                        welltype="reaction",
                        wellindex=indexwellavailable - 1,
                        volume=nextaddactionobj.volume,
                        smiles=productobj.smiles,
                        concentration=nextaddactionobj.concentration,
                        solvent=nextaddactionobj.solvent,
                        reactantfornextstep=True,
                    )

                else:
                    self.createWellModel(
                        plateobj=plateobj,
                        reactionobj=reactionobj,
                        welltype="reaction",
                        wellindex=indexwellavailable - 1,
                        smiles=productobj.smiles,
                    )

    def combinestrings(self, row):
        return (
            str(row["smiles"])
            + "-"
            + str(row["solvent"])
            + "-"
            + str(row["concentration"])
        )

    def cloneInputPlate(self, platesforcloning: list[Plate]):
        """Clones plates

        Parameters
        ----------
        platesforcloning: list[Plate]
            List of plates to be cloned
        """
        for plateobj in platesforcloning:
            indexslot = self.checkDeckSlotAvailable()
            if indexslot:
                clonewellqueryset = self.getCloneWells(plateobj=plateobj)
                plateindex = indexslot
                previousname = plateobj.name
                platename = "Startingplate"
                plateobj.pk = None
                plateobj.deck_id = self.deckobj
                plateobj.otsession_id = self.otsessionobj
                plateobj.index = plateindex
                plateobj.name = "{}_{}_from_{}".format(
                    platename, indexslot, previousname
                )
                plateobj.save()
                self.cloneInputWells(clonewellqueryset, plateobj)
            else:
                print("No more deck slots available")

    def cloneInputWells(self, clonewellqueryset: QuerySet[Well], plateobj: Plate):
        """Clones wells

        Parameters
        ----------
        clonewellqueryset: QuerySet[Well]
            The wells to be cloned
        plateobj: Plate
            The plate object (Usually previously cloned) related to the cloned well
        """
        for clonewellobj in clonewellqueryset:
            clonewellobj.pk = None
            clonewellobj.plate_id = plateobj
            clonewellobj.otsession_id = self.otsessionobj
            clonewellobj.save()

    def createSolventPlate(self, materialsdf: DataFrame):
        """Creates solvent plate/s for diluting reactants for reactions or analysis."""
        if not materialsdf.empty:
            solventdictslist = []
            materialsdf = materialsdf.groupby(["solvent"])["volume"].sum().to_frame()
            startinglabwareplatetype = self.getPlateType(platetype="solvent", volumes=materialsdf["volume"])
            # startinglabwareplatetype = self.getStartingMaterialPlateType(
            #     platetype="solvent",
            #     materialsdf=materialsdf
            # )

            plateobj = self.createPlateModel(
                platetype="solvent",
                platename="Solventplate",
                labwaretype=startinglabwareplatetype,
            )

            maxwellvolume = self.getMaxWellVolume(plateobj=plateobj)
            deadvolume = self.getDeadVolume(maxwellvolume=maxwellvolume)

            for solventgroup in materialsdf.index.values:
                totalvolume = materialsdf.at[solventgroup, "volume"]
                if totalvolume > maxwellvolume:
                    nowellsneededratio = totalvolume / (maxwellvolume - deadvolume)
                    frac, whole = math.modf(nowellsneededratio)
                    volumestoadd = [maxwellvolume for i in range(int(whole))]
                    volumestoadd.append(frac * maxwellvolume + deadvolume)

                    for volumetoadd in volumestoadd:
                        indexwellavailable = self.checkPlateWellsAvailable(
                            plateobj=plateobj
                        )
                        if not indexwellavailable:

                            plateobj = self.createPlateModel(
                                platetype="solvent",
                                platename="Solventplate",
                                labwaretype=startinglabwareplatetype,
                            )

                            indexwellavailable = self.checkPlateWellsAvailable(
                                plateobj=plateobj
                            )

                        wellobj = self.createWellModel(
                            plateobj=plateobj,
                            welltype="solvent",
                            wellindex=indexwellavailable - 1,
                            volume=volumetoadd,
                            solvent=solventgroup,
                        )

                        solventdictslist.append(
                            {
                                "name": plateobj.name,
                                "labware": plateobj.labware,
                                "well": wellobj.index,
                                "solvent": solventgroup,
                                "amount-ul": volumetoadd,
                            }
                        )

                else:
                    indexwellavailable = self.checkPlateWellsAvailable(
                        plateobj=plateobj
                    )
                    volumetoadd = totalvolume + deadvolume

                    if not indexwellavailable:
                        plateobj = self.createPlateModel(
                            platetype="solvent",
                            platename="Solventplate",
                            labwaretype=startinglabwareplatetype,
                        )
                        indexwellavailable = self.checkPlateWellsAvailable(
                            plateobj=plateobj
                        )

                    wellobj = self.createWellModel(
                        plateobj=plateobj,
                        welltype="solvent",
                        wellindex=indexwellavailable - 1,
                        volume=volumetoadd,
                        solvent=solventgroup,
                    )

                    solventdictslist.append(
                        {
                            "name": plateobj.name,
                            "labware": plateobj.labware,
                            "well": wellobj.index,
                            "solvent": solventgroup,
                            "amount-ul": volumetoadd,
                        }
                    )

            solventdf = pd.DataFrame(solventdictslist)

            self.createSolventPrepModel(solventdf=solventdf)
