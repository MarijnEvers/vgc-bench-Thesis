from poke_env.player import Player
from poke_env import RandomPlayer
from poke_env.data import GenData
import asyncio
from poke_env.battle import (
    AbstractBattle,
    DoubleBattle,
    Effect,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    PokemonGender,
    PokemonType,
    SideCondition,
    Status,
    Target,
    Weather,
)
from poke_env.environment import DoublesEnv
from poke_env.environment.env import _EnvPlayer
from poke_env.player import BattleOrder, DefaultBattleOrder, SimpleHeuristicsPlayer
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz
from vgc_bench import PolicyPlayer, BatchPolicyPlayer



team_1 = """
Dragonite @ Loaded Dice  
Ability: Multiscale  
Level: 100  
Tera Type: Fairy  
EVs: 4 HP / 252 Atk / 252 Spe  
Adamant Nature  
- Scale Shot  
- Ice Spinner  
- Tailwind  
- Protect  

Arcanine-Hisui @ Choice Band  
Ability: Intimidate  
Level: 100  
Tera Type: Ghost  
EVs: 4 HP / 252 Atk / 252 Spe  
Jolly Nature  
IVs: 25 SpA  
- Flare Blitz  
- Extreme Speed  
- Rock Slide  
- Head Smash  

Ursaluna-Bloodmoon @ Assault Vest  
Ability: Mind's Eye  
Level: 100  
Tera Type: Poison  
EVs: 156 HP / 12 Def / 252 SpA / 52 SpD / 36 Spe  
Modest Nature  
IVs: 4 Atk  
- Blood Moon  
- Hyper Voice  
- Vacuum Wave  
- Earth Power  

Sinistcha-Masterpiece @ Mental Herb  
Ability: Hospitality  
Level: 100  
Tera Type: Water  
EVs: 252 HP / 60 Def / 4 SpA / 188 SpD  
Calm Nature  
IVs: 21 Atk  
- Matcha Gotcha  
- Rage Powder  
- Strength Sap  
- Trick Room  

Sneasler @ Focus Sash  
Ability: Unburden  
Level: 100  
Tera Type: Fairy  
EVs: 4 HP / 252 Atk / 252 Spe  
Adamant Nature  
IVs: 26 SpA  
- Dire Claw  
- Close Combat  
- Tera Blast  
- Protect  

Kingambit @ Safety Goggles  
Ability: Defiant  
Level: 100  
Tera Type: Water  
EVs: 132 HP / 252 Atk / 4 Def / 84 SpD / 36 Spe  
Adamant Nature  
- Kowtow Cleave  
- Sucker Punch  
- Swords Dance  
- Protect  
"""

team_2 = """
Rillaboom @ Assault Vest  
Ability: Grassy Surge  
Level: 100  
Tera Type: Fire  
EVs: 140 HP / 196 Atk / 12 Def / 156 SpD / 4 Spe  
Adamant Nature  
- Wood Hammer  
- Grassy Glide  
- High Horsepower  
- Fake Out  

Ursaluna-Bloodmoon @ Leftovers  
Ability: Mind's Eye  
Level: 100  
Tera Type: Electric  
EVs: 164 HP / 12 Def / 36 SpA / 140 SpD / 156 Spe  
Modest Nature  
IVs: 0 Atk  
- Blood Moon  
- Earth Power  
- Yawn  
- Protect  

Ninetales-Alola @ Focus Sash  
Ability: Snow Warning  
Level: 100  
Tera Type: Ghost  
EVs: 4 HP / 252 SpA / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Blizzard  
- Encore  
- Disable  
- Protect  

Volcarona @ Grassy Seed  
Ability: Flame Body  
Level: 100  
Tera Type: Fairy  
EVs: 252 HP / 76 Def / 36 SpA / 4 SpD / 140 Spe  
Modest Nature  
- Heat Wave  
- Tera Blast  
- Quiver Dance  
- Protect  

Basculegion (M) @ Choice Scarf  
Ability: Adaptability  
Level: 100  
Tera Type: Water  
EVs: 244 Atk / 12 Def / 252 Spe  
Jolly Nature  
- Wave Crash  
- Flip Turn  
- Last Respects  
- Aqua Jet  

Incineroar @ Safety Goggles  
Ability: Intimidate  
Level: 100  
Tera Type: Ghost  
EVs: 244 HP / 4 Atk / 28 Def / 36 SpD / 196 Spe  
Careful Nature  
- Knock Off  
- Taunt  
- Parting Shot  
- Fake Out  
"""


samplelist = []
labels = []
mons = []
moves = []

#singles MaxBasePowerPlayer
#save current maxbp player, but change into tree player
class MaxBasePowerPlayer(Player):
    #put into init+assign to self
    pokemon = [x.splitlines() for x in team_1.split("\n\n")]
    movesr = [[l for l in p if l.startswith("- ")] for p in pokemon]


    for p in pokemon:
        for a in p:
            if " @ " in a:
                mons.append(a.split("@")[0].lower().replace("-", "").replace(" ","")) 
    
    for mon in movesr:
        mlist = []
        for m in mon:
            mlist.append(m.split("-")[1].lower().replace(" ", ""))
        moves.append(mlist)


    print(mons)

    print(moves)

    #can get things from self, use the super to extend simpleheuristic player
    #get best_move index from moves, for better tree,+labeling?
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)

            if battle.can_tera:
                return self.create_order(best_move, terastallize=True)
            
            
            attributelist = [] 
            
            active_mon = battle.active_pokemon.species.lower() #current own active pokemon, pokemon.name removes distinctions between forms, so use pokemon.species
            print(active_mon)

            active_index = mons.index(active_mon) #index of own active mon

            attributelist = [-1 for i in range(4)]

            for move in battle.available_moves:
                if str(best_move).split()[0] != "struggle":
                    i = moves[active_index].index(str(best_move).split()[0])
                    attributelist[i] = move.base_power
                

            
            print(battle.available_moves[0].id)
            #for x in range(0,3): #doesnt work, due to available moves only length 4 if nothing is happening. create addtional index variable for available moves?
            #    if moves[active_index][x] != battle.available_moves[x].id:
            #        attributelist.insert(x, 0)

            #scuffed fixing of attributelist length incases of choicelock/encore etc, such that tree construction still works
            #for x in range(1,4):
            #    if len(attributelist) <= x:
            #        print("less then 4: ", attributelist)
            #        attributelist.append(0)


            samplelist.append(attributelist)
            if str(best_move).split()[0] == "struggle":
                labels.append(5)
            else:
                labels.append(6 * active_index + moves[active_index].index(str(best_move).split()[0]))
            print(attributelist)
            print(".") 
            print(best_move._id) #class label
            print(battle.available_moves.index(best_move))
            print("next turn")

            

            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)

class HeuristicPlayer(SimpleHeuristicsPlayer):
    # look *args and *kwargs

    #def __init__(self, account_configuration = None, *, avatar = None, battle_format, log_level = None, max_concurrent_battles = 1, accept_open_team_sheet = False, save_replays = False, server_configuration = ..., start_timer_on_battle_start = False, start_listening = True, open_timeout = 10, ping_interval = 20, ping_timeout = 20, loop = ..., team = None, strict_battle_tracking = False):
     #   super().__init__(account_configuration, avatar=avatar, battle_format=battle_format, log_level=log_level, max_concurrent_battles=max_concurrent_battles, accept_open_team_sheet=accept_open_team_sheet, save_replays=save_replays, server_configuration=server_configuration, start_timer_on_battle_start=start_timer_on_battle_start, start_listening=start_listening, open_timeout=open_timeout, ping_interval=ping_interval, ping_timeout=ping_timeout, loop=loop, team=team, strict_battle_tracking=strict_battle_tracking)

    def choose_move(self, battle):
        chosen_move = super().choose_move(battle) # type: ignore
        print(type(chosen_move))
        print(chosen_move)

        attributelist = [] 
    


        active_mon = battle.active_pokemon.species.lower() #current own active pokemon, pokemon.name removes distinctions between forms, so use pokemon.species
        print(active_mon)

        active_index = mons.index(active_mon) #index of own active mon


        #rename attributelist? into state attributes?
        #attribute list
        #global: terrain, weather, gravity, tr, wonder room, magic room, neutralizing gas
        #one side: tw, safeguard, mist, rainbow, swamp, sea of fire, HAZARDS:spikes, toxic spikes, stealth rock, web, SCREENS: light screen, reflect, Aurora veil, 
        #either one side or one pokemon?, trapped
        #one pokemon: non volatile status, stat changes, perish counter, cursed, disable, encore, outrage lock, semi invuln turn, prot move last turn, choice locked, confused, drowsy
        #pokemon specific: typing, abil(how?), item known moves(?)
        #moves: type, bp, acc, category, 2ndary effect, 2ndary effect chance, pp left, targeting   (how to do status moves exactly?)



        #change for more move attributes
        #4*x, where x is ammount of attributes
        #attribute: bp, acc, move type(phys/spec/status), typing?, 

        #global attributes, how to do duration/turns left?
        
        print(battle.fields) #every global field effect except weather, prints which is active, terrain is mut exclu, rest isnt
        #terrains
        if Field.GRASSY_TERRAIN in battle.fields:
            print("grass")
        elif Field.ELECTRIC_TERRAIN in battle.fields:
            print("electric")
        elif Field.MISTY_TERRAIN in battle.fields:
            print("misty")
        elif Field.PSYCHIC_TERRAIN in battle.fields:
            print("psychic")

        #other global attributes
        if Field.GRAVITY in battle.fields:
            print("gravity")
        if Field.NEUTRALIZING_GAS in battle.fields:
            print("neutral gas")
        if Field.TRICK_ROOM in battle.fields:
            print("tr") 
        
        
        print(battle.weather) #weather, ignore primal weather, not possible in current formats? weather is ut exclu
        if Weather.SNOWSCAPE in battle.weather or Weather.SNOW in battle.weather:
            print("snow")
        elif Weather.SANDSTORM in battle.weather:
            print("sand")
        elif Weather.RAINDANCE in battle.weather:
            print("rain")
        elif Weather.SUNNYDAY in battle.weather:
            print("sun")


        #library for what attribute is where
        attributelist = [-1 for i in range(8)]

        
        for move in battle.available_moves: #states
            if str(chosen_move).split()[2] != "struggle" and str(chosen_move).split()[1] != "switch":
                i = moves[active_index].index(str(move).split()[0])
                attributelist[2 * i] = move.base_power
                attributelist[2 * i + 1] = move.accuracy
                #attributelist[3 * i + 2] = move.




        print(battle.available_moves)

        
        samplelist.append(attributelist)
        if str(chosen_move).split()[2] == "struggle":
            labels.append(30)
        elif str(chosen_move).split()[1] == "switch":
            labels.append(active_index + 24)
        else:
            labels.append(active_index * 4 + moves[active_index].index(str(chosen_move).split()[2]))
        
        print(attributelist)
        print(".") 
        #print(chosen_move._id) #class label
        #print(battle.available_moves.index(str(chosen_move).split()[2]))
        print("next turn")
        return(chosen_move)


#player.py MaxBasePowerPlayer and SimpleHeuristicPayer work for both singles and doubles 


class MLPlayer(PolicyPlayer):
    def choose_move(self, battle):
        chosen_move = super().choose_move(battle) # type: ignore
        print(type(chosen_move))
        print(chosen_move)

        attributelist = [] 
    


        active_mon = battle.active_pokemon.species.lower() #current own active pokemon, pokemon.name removes distinctions between forms, so use pokemon.species
        print(active_mon)

        active_index = mons.index(active_mon) #index of own active mon


        #rename attributelist? into state attributes?
        #attribute list
        #global: terrain, weather, gravity, tr, wonder room, magic room
        #one side: tw, safeguard, mist, rainbow, swamp, sea of fire, HAZARDS:spikes, toxic spikes, stealth rock, web, SCREENS: light screen, reflect, Aurora veil, 
        #one pokemon: non volatile status, stat changes, perish counter, cursed, disable, encore, outrage lock, semi invuln turn, prot move last turn, choice locked
        #pokemon specific: typing, abil(how?), item known moves(?)
        #moves: type, bp, acc, category, 2ndary effect, 2ndary effect chance, pp left, targeting   (how to do status moves exactly?)



        #change for more move attributes
        #4*x, where x is ammount of attributes
        #attribute: bp, acc, move type(phys/spec/status), typing?, 


        # i mod 3 = 0 -> bp, i mod 3 = 1 -> acc, i mod 3 = 2 -> move category
        attributelist = [-1 for i in range(8)]
        
        for move in battle.available_moves: #states
            if str(chosen_move).split()[2] != "struggle" and str(chosen_move).split()[1] != "switch":
                i = moves[active_index].index(str(move).split()[0])
                attributelist[2 * i] = move.base_power
                attributelist[2 * i + 1] = move.accuracy
                #attributelist[3 * i + 2] = move.




        print(battle.available_moves)

        
        samplelist.append(attributelist)
        if str(chosen_move).split()[2] == "struggle":
            labels.append(30)
        elif str(chosen_move).split()[1] == "switch":
            labels.append(active_index + 24)
        else:
            labels.append(active_index * 4 + moves[active_index].index(str(chosen_move).split()[2]))
        
        print(attributelist)
        print(".") 
        #print(chosen_move._id) #class label
        #print(battle.available_moves.index(str(chosen_move).split()[2]))
        print("next turn")
        return(chosen_move)



p1 = HeuristicPlayer(battle_format="gen9anythinggoes", team=team_1)
p2 = RandomPlayer(battle_format="gen9anythinggoes", team=team_2)


async def main():
    print("start battle")
    await p1.battle_against(p2, n_battles=10)
    print("battle done")
    features = np.array(samplelist)
    classes = np.array(labels)
    print(features)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, classes)
    #tree.plot_tree(clf, class_names=list(map(str, range(4))), filled=True, fontsize=10)
    flatmoves = [element for innerList in moves for element in innerList]
    for mon in mons:
        flatmoves.append("switch" + mon)
    flatmoves.append("struggle")
    tree.plot_tree(clf, class_names=list(map(str, flatmoves)), filled=True, fontsize=10)
    plt.show()
    r = tree.export_text(clf) #look at labeling feature names in exported rules
    print(r)


asyncio.run(main())