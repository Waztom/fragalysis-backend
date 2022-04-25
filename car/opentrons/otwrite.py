# this is vunrable to python injection by the lack of checking of metadata inputs
# this opens and closes files frequently, could be improved by creating string to hold the file data before writing to file once

#   NOTE: in this file "humanread" referes to the comments above each line/set of lines of ot code, human readable is a list of all the comments in format [oporator (human/ot), comment]
"""Create otScript session"""
from __future__ import annotations
from django.core.files.storage import default_storage
from django.conf import settings

import os

from car.models import Batch, Reaction

from car.models import (
    Product,
    Pipette,
    TipRack,
    Plate,
    Well,
    OTScript,
)


import math


class otWrite(object):
    """ "
    Creates a otScript object for generating an OT protocol
    script
    """

    def __init__(
        self,
        protocolname: str,
        otsessionobj: Django_object,
        alladdactionsquerysetflat: list,
        apiLevel="2.9",
        reactionplatequeryset: list = None,
    ):
        self.reactionstep = otsessionobj.reactionstep
        self.otsessionobj = otsessionobj
        self.otsessionid = otsessionobj.id
        self.alladdactionsquerysetflat = alladdactionsquerysetflat
        self.platequeryset = self.getPlates()
        self.tiprackqueryset = self.getTipRacks()
        self.pipetteobj = self.getPipette()
        self.pipettename = self.pipetteobj.pipettename
        self.protocolname = protocolname
        self.filepath, self.filename = self.createFilePath()
        self.apiLevel = apiLevel
        self.reactionplatequeryset = reactionplatequeryset
        self.setupScript()
        self.setupPlates()
        self.setupTipRacks()
        self.setupPipettes()
        self.writeAddActions()
        self.createOTScriptModel()

    def getProduct(self, reaction_id):
        productobj = Product.objects.get(reaction_id=reaction_id)
        return productobj

    def getProductQuerySet(self, reaction_id: int):
        """Get product queryset for reaction_id
        Args:
            reaction_id (int): Method DB id to search for
        Returns:
            productqueryset (Django_queryset): Reaction queryset related to method_id
        """
        productqueryset = Product.objects.filter(reaction_id=reaction_id)
        return productqueryset

    def getReaction(self, reaction_id: int):
        """Get reaction object
        Args:
            reaction_id (int): Reaction DB id to search for
        Returns:
            reactionobj (Django_obj): Reaction Django object
        """
        reactionobj = Reaction.objects.get(id=reaction_id)
        return reactionobj

    def getPlates(self):
        platequeryset = Plate.objects.filter(otsession_id=self.otsessionid).order_by(
            "id"
        )
        return platequeryset

    def getPlateObj(self, plateid):
        plateobj = Plate.objects.filter(id=plateid)[0]
        return plateobj

    def getTipRacks(self):
        tipracksqueryset = TipRack.objects.filter(
            otsession_id=self.otsessionid
        ).order_by("id")
        return tipracksqueryset

    def getPipette(self):
        pipetteobj = Pipette.objects.filter(otsession_id=self.otsessionid)[0]
        return pipetteobj

    def getProductSmiles(self, reactionid):
        productobj = Product.objects.filter(reaction_id=reactionid)[0]
        return productobj.smiles

    def getPreviousObjEntry(self, queryset: list, obj: Django_obj):
        """Finds previous object relative to obj of queryset"""
        previousobj = queryset.filter(pk__lt=obj.pk).order_by("-pk").first()
        return previousobj

    def getPreviousObjEntries(self, queryset: list, obj: DjangoObjectType):
        """Finds all previous objects relative to obj of queryset"""
        previousqueryset = queryset.filter(pk__lt=obj.pk).order_by("-pk")
        return previousqueryset

    def getReactionQuerySet(self, method_id: int):
        """Get product queryset for reaction_id
        Args:
            method_id (int): Method DB id to search for
        Returns:
            reactionqueryset (Django_queryset): Reaction queryset related to method_id
        """
        reactionqueryset = Reaction.objects.filter(method_id=method_id)
        return reactionqueryset

    def checkPreviousReactionProducts(self, reaction_id: int, smiles: str):
        """Checks if any previous reactions had the product mathcing the smiles"""

        reactionobj = self.getReaction(reaction_id=reaction_id)
        reactionqueryset = self.getReactionQuerySet(method_id=reactionobj.method_id.id)
        prevreactionqueryset = self.getPreviousObjEntries(
            queryset=reactionqueryset, obj=reactionobj
        )
        if prevreactionqueryset:
            for reactionobj in prevreactionqueryset:
                productobj = self.getProduct(reaction_id=reactionobj)
                if productobj.smiles == smiles:
                    return True, reactionobj
                else:
                    return False, None
        else:
            return False, None

    # def checkPreviousReactionProduct(self, reaction_id: int, smiles: str):
    #     """Checks if the
    #     """
    #     reactionobj = self.getReaction(reaction_id=reaction_id)
    #     reactionqueryset = self.getReactionQuerySet(method_id=reactionobj.method_id)
    #     prevreactionobj = self.getPreviousObjEntry(
    #         queryset=reactionqueryset, obj=reactionobj
    #     )
    #     if prevreactionobj:
    #         prevproductobj = self.getProduct(reaction_id=prevreactionobj.id)
    #         if prevproductobj.smiles == smiles:
    #             return True, prevreactionobj
    #         else:
    #             return False, prevreactionobj
    #     else:
    #         return False, prevreactionobj

    def createFilePath(self):
        filename = "ot-script-batch-{}-reactionstep{}-sessionid-{}.txt".format(
            self.protocolname, self.reactionstep, self.otsessionid
        )
        path = "tmp/" + filename
        filepath = str(os.path.join(settings.MEDIA_ROOT, path))
        return filepath, filename

    def createOTScriptModel(self):
        otscriptobj = OTScript()
        otscriptobj.otsession_id = self.otsessionobj
        otscriptfile = open(self.filepath, "rb")
        otscriptfn = default_storage.save(
            "otscripts/{}.py".format(self.filename.strip(".txt")), otscriptfile
        )
        otscriptobj.otscript = otscriptfn
        otscriptobj.save()

    def findSolventPlateWellObj(self, solvent, transfervolume):
        """Finds solvent well for diluting a previous reaction stesp product"""
        wellinfo = []
        try:
            wellobjects = Well.objects.filter(
                otsession_id=self.otsessionid,
                solvent=solvent,
                available=True,
            ).order_by("id")
            for wellobj in wellobjects:
                areclose = self.checkVolumeClose(volume1=transfervolume, volume2=0.00)
                if areclose:
                    break
                wellvolumeavailable = self.getWellVolumeAvailable(wellobj=wellobj)
                if wellvolumeavailable > 0:
                    if wellvolumeavailable > transfervolume:
                        self.updateWellVolume(
                            wellobj=wellobj, transfervolume=transfervolume
                        )
                        wellinfo.append([wellobj, transfervolume])
                        transfervolume = 0.00
                    if wellvolumeavailable < transfervolume:
                        self.updateWellVolume(
                            wellobj=wellobj, transfervolume=wellvolumeavailable
                        )
                        wellinfo.append([wellobj, wellvolumeavailable])
                        transfervolume = transfervolume - wellvolumeavailable
        except Exception as e:
            print(e)
            print(solvent)
        return wellinfo

    def findStartingPlateWellObj(
        self, reactionid, smiles, solvent, concentration, transfervolume
    ):
        isproduct, previousreactionobj = self.checkPreviousReactionProducts(
            reaction_id=reactionid, smiles=smiles
        )
        wellinfo = []
        if isproduct:
            wellobj = Well.objects.get(
                otsession_id=self.otsessionid,
                reaction_id=previousreactionobj,
                smiles=smiles,
            )
            wellinfo.append([isproduct, wellobj, transfervolume])
        else:
            try:
                wellobjects = Well.objects.filter(
                    otsession_id=self.otsessionid,
                    smiles=smiles,
                    solvent=solvent,
                    concentration=concentration,
                    available=True,
                ).order_by("id")
                for wellobj in wellobjects:
                    areclose = self.checkVolumeClose(
                        volume1=transfervolume, volume2=0.00
                    )
                    if areclose:
                        break
                    wellvolumeavailable = self.getWellVolumeAvailable(wellobj=wellobj)
                    if wellvolumeavailable > 0:
                        if wellvolumeavailable > transfervolume:
                            self.updateWellVolume(
                                wellobj=wellobj, transfervolume=transfervolume
                            )
                            wellinfo.append([isproduct, wellobj, transfervolume])
                            transfervolume = 0.00
                        if wellvolumeavailable < transfervolume:
                            self.updateWellVolume(
                                wellobj=wellobj, transfervolume=wellvolumeavailable
                            )
                            wellinfo.append([isproduct, wellobj, wellvolumeavailable])
                            transfervolume = transfervolume - wellvolumeavailable
            except Exception as e:
                print(e)
                print(smiles, solvent, concentration)

        return wellinfo

    def checkVolumeClose(self, volume1, volume2):
        checkclose = math.isclose(volume1, volume2, rel_tol=0.001)
        if checkclose:
            return True
        else:
            return False

    def findReactionPlateWellObj(self, reactionid):
        productsmiles = self.getProductSmiles(reactionid=reactionid)
        wellobj = Well.objects.filter(
            otsession_id=self.otsessionid, reaction_id=reactionid, smiles=productsmiles
        )[0]
        return wellobj

    def getWellVolumeAvailable(self, wellobj):
        plateid = wellobj.plate_id.id
        maxwellvolume = self.getMaxWellVolume(plateid=plateid)
        deadvolume = self.getDeadVolume(maxwellvolume=maxwellvolume)
        wellvolume = wellobj.volume
        wellvolumeavailable = wellvolume - deadvolume
        self.updateWellAvailable(
            wellvolumeavailable=wellvolumeavailable, wellobj=wellobj
        )
        return wellvolumeavailable

    def updateWellAvailable(self, wellvolumeavailable, wellobj):
        if wellvolumeavailable < 0:
            wellobj.available = False
            wellobj.save()

    def updateWellVolume(self, wellobj, transfervolume):
        wellobj.volume = wellobj.volume - transfervolume
        wellobj.save()

    def getMaxWellVolume(self, plateid):
        plateobj = self.getPlateObj(plateid)
        maxwellvolume = plateobj.maxwellvolume
        return maxwellvolume

    def getDeadVolume(self, maxwellvolume):
        deadvolume = maxwellvolume * 0.05
        return deadvolume

    def setupScript(self):
        """This is vunrable to injection atacks """
        script = open(self.filepath, "w")
        script.write("from opentrons import protocol_api\n")
        script.write(
            "# "
            + str(self.protocolname)
            + str('" produced by XChem Car (https://car.xchem.diamond.ac.uk)')
        )
        script.write("\n# metadata")
        script.write(
            "\nmetadata = {'protocolName': '"
            + str(self.protocolname)
            + "','apiLevel': '"
            + str(self.apiLevel)
            + "'}\n"
        )
        script.write("\ndef run(protocol: protocol_api.ProtocolContext):\n")

        script.close()

    def setupPlates(self):
        script = open(self.filepath, "a")
        script.write("\n\t# labware")
        for plateobj in self.platequeryset:
            platename = plateobj.platename
            labware = plateobj.labware
            plateindex = plateobj.plateindex
            script.write(
                f"\n\t{platename} = protocol.load_labware('{labware}', '{plateindex}')"
            )

        script.close()

    def setupTipRacks(self):
        script = open(self.filepath, "a")
        for tiprackobj in self.tiprackqueryset:
            tiprackname = tiprackobj.tiprackname
            labware = tiprackobj.labware
            tiprackindex = tiprackobj.tiprackindex
            script.write(
                f"\n\t{tiprackname} = protocol.load_labware('{labware}', '{tiprackindex}')"
            )

        script.close()

    def setupPipettes(self):
        script = open(self.filepath, "a")
        script.write("\n\n\t# pipettes\n")
        script.write(
            "\t"
            + str(self.pipetteobj.pipettename)
            + " = protocol.load_instrument('"
            + str(self.pipetteobj.labware)
            + "', '"
            + str(self.pipetteobj.position)
            + "', tip_racks=["
            + ",".join([tiprackobj.tiprackname for tiprackobj in self.tiprackqueryset])
            + "])\n"
        )

        script.close()

    def writeCommand(self, comandString):
        script = open(self.filepath, "a")
        if type(comandString) == str:
            script.write("\t" + str(comandString) + "\n")
        elif type(comandString) == list:
            for command in comandString:
                script.write("\t" + str(command) + "\n")

        script.close()

    def mixWell(self, wellindex: int, nomixes: int, plate: str, volumetomix: float):
        """Mixes conents of well"""
        humanread = f"Mixing contents of well at index: {wellindex}"

        instruction = [
            "\n\t# " + str(humanread),
            self.pipettename
            + f".mix({nomixes}, {volumetomix}, {plate}.wells([{wellindex}])",
        ]

        self.writeCommand(instruction)

    def transferFluid(
        self,
        fromplatename,
        toplatename,
        fromwellindex,
        towellindex,
        transvolume,
        takeheight=2,
        dispenseheight=-5,
    ):
        humanread = (
            f"transfer - {transvolume:.1f}ul from {fromwellindex} to {towellindex}"
        )

        instruction = [
            "\n\t# " + str(humanread),
            self.pipettename
            + f".transfer({transvolume}, {fromplatename}.wells()[{fromwellindex}].bottom({takeheight}), {toplatename}.wells()[{towellindex}].top({dispenseheight}), air_gap = 15)",
        ]

        self.writeCommand(instruction)

    def pickUpTip(self):
        humanread = "Pick up tip"

        instruction = [
            "\n\t# " + str(humanread),
            self.pipettename + ".pick_up_tip()",
        ]

        self.writeCommand(instruction)

    def disposeTip(self):
        humanread = "Dispose tip"

        instruction = [
            "\n\t# " + str(humanread),
            self.pipettename + ".drop_tip()",
        ]

        self.writeCommand(instruction)

    def writeAddActions(self):
        for addaction in self.alladdactionsquerysetflat:
            transfervolume = addaction.materialquantity
            solvent = addaction.solvent

            fromwellinfo = self.findStartingPlateWellObj(
                reactionid=addaction.reaction_id.id,
                smiles=addaction.materialsmiles,
                solvent=solvent,
                concentration=addaction.concentration,
                transfervolume=transfervolume,
            )

            for wellinfo in fromwellinfo:
                isproduct = wellinfo[0]
                fromwellobj = wellinfo[1]
                transfervolume = wellinfo[2]

                if isproduct:
                    # Add dilution
                    fromsolventwellinfo = self.findSolventPlateWellObj(
                        solvent=solvent,
                        transfervolume=transfervolume,
                    )
                    for solventwellinfo in fromsolventwellinfo:
                        fromwellobj = solventwellinfo[0]
                        transfervolume = solventwellinfo[1]
                        towellobj = fromwellobj
                        fromplateobj = self.getPlateObj(plateid=fromwellobj.plate_id.id)
                        toplateobj = self.getPlateObj(plateid=towellobj.plate_id.id)

                        fromplatename = fromplateobj.platename
                        toplatename = toplateobj.platename
                        fromwellindex = fromwellobj.wellindex
                        towellindex = towellobj.wellindex

                        self.transferFluid(
                            fromplatename=fromplatename,
                            toplatename=toplatename,
                            fromwellindex=fromwellindex,
                            towellindex=towellindex,
                            transvolume=transfervolume,
                        )

                    self.mixWell(
                        wellindex=towellindex,
                        nomixes=3,
                        plate=toplatename,
                        volumetomix=transfervolume,
                    )

                towellobj = self.findReactionPlateWellObj(
                    reactionid=addaction.reaction_id.id
                )
                fromplateobj = self.getPlateObj(plateid=fromwellobj.plate_id.id)
                toplateobj = self.getPlateObj(plateid=towellobj.plate_id.id)

                fromplatename = fromplateobj.platename
                toplatename = toplateobj.platename
                fromwellindex = fromwellobj.wellindex
                towellindex = towellobj.wellindex

                self.transferFluid(
                    fromplatename=fromplatename,
                    toplatename=toplatename,
                    fromwellindex=fromwellindex,
                    towellindex=towellindex,
                    transvolume=transfervolume,
                )
