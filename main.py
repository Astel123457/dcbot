import discord
from discord import app_commands
import asyncio
from discord.ext import commands
from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage
import random
import torch
from torchvision import models
import torchvision.transforms as transforms
import requests
import catbox
import io
import numpy as np
import base64
from PIL import Image, ImageEnhance, ImageOps, ImageSequence
import json
import traceback
import time
import chess
import chess.engine
from pixelsort import pixelsort
import aiohttp
import ffmpeg
import os
import cairosvg
from io import BytesIO
import memory_text
from tenacity import retry, stop_after_attempt, retry_if_exception
import bs4
import urllib
import clientsecrets as cs
from music_cog import Music

#init intents and dicord client
intents = discord.Intents.default()
intents.message_content = True

activity = discord.Activity(name='you poop, smells nice >:3', type=discord.ActivityType.watching)
#why does discord do this to me
bot = commands.Bot(command_prefix='!', intents=intents, activity=activity)
bot.remove_command('help')

token = cs.catbox_key

uploader = catbox.Uploader(token)

#init mistral client

api_key = cs.mistral_key
model = "mistral-large-latest"

stop = False

client = MistralAsyncClient(api_key=api_key)

#list of engines and their location as well as storing games
engines = {
    "stockfish": "stockfish-windows-x86-64-avx2.exe",
    "lc0": "lc0.exe",
    "komodo": "komodo-14.1-64bit-bmi2.exe"
}

games = {}

#all this is for Stable Diffusion

sd_quality = {}

curr_style = {"default": "stable-diffusion"}

STYLE_LIST = {}

last_message = ""

last_sent_message = None

temp = 7

negitive_prompt = "lowres, ((bad_anatomy)), ((bad_hands)), text, missing_finger, extra_digits, fewer_digits, blurry, ((mutated hands and fingers)), (poorly drawn face), ((mutation)), ((deformed_face)), (ugly), ((bad_proportions)), ((extra_limbs)), extra_face, (double_head), (extra_head), ((extra_feet)), monster, logo, cropped, worst_quality, low_quality, normal_quality, jpeg, long_body, long_neck, ((jpeg artifacts))"

tiling = {}

with open("styles.json", "r") as file:
    STYLE_LIST = json.load(file)
#load chat memory
try:         
    message_history = memory_text.ChatMemory.load()
except:
    message_history = memory_text.ChatMemory(max_message_limit=100)
#get a set of labels from imagenet for def rn()
url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

class_labels = urllib.request.urlopen(url).read().decode()
class_labels = class_labels.split("\n")

# Load a pre-trained ResNet model for rn
resnet = models.resnet18(pretrained=True)

#Get copypastas from r74n.com
r = requests.get("https://c.r74n.com/copypastas")

txt = r.content

soup = bs4.BeautifulSoup(txt, "html.parser")
soup = soup.find_all("textarea")

soup_clean = []
for i in soup:
    i = i.text
    i.replace("<textarea>", "")
    i.replace("</textarea>", "")
    soup_clean.append(i)

#make a bag of copypastas

class Bag:
    def __init__(self, items):
        self.items = items
        self.bag = []

    def shuffle(self):
        self.bag = random.sample(self.items, len(self.items))

    def draw(self):
        if not self.bag:
            self.shuffle()
        return self.bag.pop()

    def reset(self):
        self.bag = self.items
    
bag = Bag(soup_clean)

#initialize responses 
responses = {
    'hello': "Hello! >:3",
    'goblin': "this dick",
    'suckon': "deez nuts",
    'candice': "dick fit in yo mouth",
    'hi': 'Hi there!',
    'deez': "nuts",
    'fuck you': "No thanks",
    'die': ":skull:",
    'kill': "*dies*",
    'live': "*resurrects*,\n\nI'm alive.",
    'ligma': "ballz"
}

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')
    print("Syncing commands...")
    await bot.tree.sync(guild=discord.Object(id=889711909379641375))
    await bot.tree.sync()
    print("Commands synced!")
    print('------')

@bot.event
async def on_message_edit(before, after):
    if after.author == bot.user:
        return
    #send back to on_message but with the is_edit flag set to true
    if len(after.embeds) > len(before.embeds):
        pass
    else:
        await on_message(after, True)

@bot.command()
async def sync(ctx):
    if ctx.author.id != 493626802644713473:
        await ctx.send("Only Astel can run this command!")
        return
    print("Syncing commands...")
    await bot.tree.sync()
    print("Commands synced!")

async def run_chat_send(queue):
    while True:
        srn, stop, ctx = await queue.get()
        if stop is not None:
            try:
                await ctx.edit(content=srn+"...")
            except Exception as e:
                await ctx.channel.send(f"oops, something went wrong while streaming {e}")
        else:
            await ctx.edit(content=srn)
            break
        await asyncio.sleep(1)

@bot.event
async def on_message(message, is_edit = False):
    #get message data and print it to the console
    global message_history
    global stop
    print("Message Recived!")
    if message.guild == None:
       print(f"Channel sent: {message.channel.id} in \"{message.channel}\"")
    else:
        print(f"Channel sent: {message.channel.id} in \"{message.channel}\" in \"{message.guild.name}\"")
    print(f"Author: {message.author}")
    print(f"Message: {message.content}")
    print("------")
    #if the message is not an edit, add one to the message count (using a file because )
    if not is_edit:
        f = open("messages.txt", "r")
        test = f.read()
        messages_sent = int(test) + 1
        f.close()
        m = open("messages.txt", "w")
        messages_sent = str(messages_sent)
        m.write(messages_sent)
        m.close()
    if message.author == bot.user:
        return
    check = message.content.lower()
    message_author = str(message.author)
    if message.content.startswith('$'):
        current = await message.reply("...")
        queue = asyncio.Queue(maxsize=1)
        task = asyncio.create_task(run_chat_send(queue))
        async with message.channel.typing():
            message_history.add(ChatMessage(role="user", content=message.content[1:]))
            total = ""
            cur = ""
            async for chunk in client.chat_stream(model=model, messages=message_history.construct_history()):  
                if chunk.choices[0].delta.content == None:
                    continue
                cur += chunk.choices[0].delta.content
                total += chunk.choices[0].delta.content
                new = False
                if chunk.choices[0].finish_reason is not None:
                    if not queue.empty():
                        queue.get_nowait()
                    await queue.put((cur, None, current))
                elif stop:
                    stop = False
                    message_history.add(ChatMessage(role="assistant", content=total+"- (Stopped by User)"))
                    if not queue.empty():
                        queue.get_nowait()
                    await queue.put((cur+"- (Stopped)", None, current))
                    await message.reply("MistralAI was responding, all chat streams have been stopped!")
                    break
                else:
                    if len(cur) > 1950:
                        cur = [cur[i:i + len(cur)] for i in range(0, len(cur), 1950)][-1]
                        current = await message.reply("...")
                    elif not queue.empty():
                        queue.get_nowait()
                    await queue.put((cur, True, current))
            message_history.add(ChatMessage(role="assistant", content=total))

    if not is_edit:
        if message.channel.id == 1083934617637232710:
            return
        else:
        # if message_author == "Astel123457#0421": 
        #    await message.channel.send("Aii my queen my love kiss me and marry me pls")
            message_first_word = message.content.split(" ")[0]
            if message_first_word.lower() in responses:
                await message.channel.send(responses[message_first_word.lower()])
    # if message.content.startswith('e'):
    #     await message.channel.send("a")
    #     await asyncio.sleep(0.5)
    #     await message.channel.send("sports")
            if message.author.name == "IDOLTRASH":
                niagarafalls = ["changed", "change"]
                thispussy = any
                thong = message.content.lower()
                thongs = message.channel.send
                if thispussy(wet in thong for wet in niagarafalls):
                    await thongs("Thongs = changed")
    is_edit = False
    await bot.process_commands(message)

@bot.tree.command(description="Spams a bunch of characters.")
async def spam(ctx):
    result_str = ''.join((random.choice('qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM,.;[]1234567890-=_+') for i in range(random.randint(0,2000))))
    await ctx.send(result_str)

@bot.tree.command(description="Generates a non cryptograpic password.")
async def secure_password(ctx):
    result_str = ''.join((random.choice('qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM,.;[]1234567890-=_+') for i in range(random.randint(6,13))))
    await ctx.send("Here is a password!")
    await ctx.send(result_str)



@bot.command()
async def timer(ctx, time = 10):
    await ctx.send(f"Starting timer for {time} seconds")
    f = open("messages.txt", "w")
    f.write("0")
    f.close()
    time = int(time)
    await asyncio.sleep(time)
    s = open("messages.txt", "r")
    messages = s.read()
    s.close()
    await ctx.send(f"There were {messages} messages in the span of {time} seconds!")

@bot.command()
async def send_message(ctx, channel_id, *, message):
    channelid = int(channel_id)
    channel = bot.get_channel(channelid)
    await channel.send(message)
    print("Message sent!")
    print("------")

def retry_if_not_specific_errors(exception):
    # Define the specific set of errors that should not trigger a retry
    specific_errors = (TypeError, ValueError)
    
    # Retry if the exception is not in the specific set of errors
    return not isinstance(exception, specific_errors)

@retry(stop=stop_after_attempt(3),retry=retry_if_exception(retry_if_not_specific_errors))
async def try_resend_file(ctx, file_objects):
    if not isinstance(file_objects, list):
        raise TypeError("file_objects must be a list of discord.File objects")
    elif len(file_objects) == 0:
        raise ValueError("file_objects must have at least one file")
    elif len(file_objects) > 10:
        raise ValueError("file_objects can have at most 10 files")
    elif not isinstance(file_objects[0], discord.File):
        raise TypeError("file_objects must be a list of discord.File objects")
    else:
        await ctx.send(files=file_objects)

async def getinf(ctx, url, headers, id, filename):
    while True:
        try:
            check = requests.get(f"{url}/result/{id}", headers=headers)
        
            if check.status_code == 200:
                print(check.status_code)
                if filename == "video.mp4":
                    try:
                        fstream = io.BytesIO(base64.b64decode(check.json()['video']))
                        fstream.seek(0)
                        await ctx.reply(f"ID: {id}", file=discord.File(fstream, filename="video.mp4"))
                    except aiohttp.client_exceptions.ClientOSError:
                        upload = uploader.upload(file_type="mp4", file_raw=fstream)
                        upload = upload["file"]
                        await ctx.reply(f"ID: {id}\n{upload}")
                else:
                    try:
                        fstream = io.BytesIO(base64.b64decode(check.json()['image']))
                        fstream.seek(0)
                        await ctx.reply(f"ID: {id}", file=discord.File(fstream, filename="image.png"))
                    except aiohttp.client_exceptions.ClientOSError:
                        with open("img.png", "wb") as file:
                            file.write(fstream.read())
                        upload = uploader.upload(file_type="png", file_raw=fstream.read())
                        upload = upload["file"]
                        print(upload)
                        await ctx.reply(f"ID: {id}\n{upload}")
                        
                break
            elif check.status_code == 403:
                await ctx.reply(f"Your image or prompt was flagged by Stability AI's content system, if you belive this was a mistake, try again.\nError Details: {check.text}")
                return
            elif check.status_code != 202:
                await ctx.reply(f"There was an error generating the video! Please try again or use `!getvid` with the ID to try getting the video again.\nID: {id}")
                break
            await asyncio.sleep(2)
        except Exception as e:
            print(traceback.format_exc())
            await ctx.reply(f"There was an error getting the video! Please try again or use `!getvid` with the ID to try getting the video again.\nID: {id}\nError Details: {e}")
            break

@bot.command()
async def getvid(ctx, id = None):
    if id == None:
        await ctx.send("You must input an id!")
        return
    headers2 = {
        'Authorization' : f"Bearer {cs.stability_key}",
        "Accept" : "application/json"
    }
    url = "https://api.stability.ai/v2beta/image-to-video"
    await getinf(ctx, url, headers2, id, "video.mp4")

@bot.command()
async def getupscale(ctx, id = None):
    if id == None:
        await ctx.send("You must input an id!")
        return
    headers2 = {
        'Authorization' : f"Bearer {cs.stability_key}",
        "Accept" : "application/json"
    }
    url = "https://api.stability.ai/v2beta/stable-image/upscale/creative"
    await getinf(ctx, url, headers2, id, "img.png")

@bot.command()
async def imgtovid(ctx, motion: int = 40):
    url = "https://api.stability.ai/v2beta/image-to-video"
    if len(ctx.message.attachments) == 0:
        await ctx.send("You must attach an image to this command!")
        return
    if len(ctx.message.attachments) > 1:
        await ctx.send("You can only attach one image at a time!")
        return
    await ctx.reply("Please note that this may take a while to generate!")
    if motion > 255 or motion < 1:
        await ctx.reply("`Motion` parameter must be between 255 and 1")
    # Assuming org_image is obtained from ctx.message.attachments[0].read()
    org_image = await ctx.message.attachments[0].read()
    img = Image.open(io.BytesIO(org_image))
    
    max_size = 768
    
    # Determine which dimension is larger and calculate new size maintaining aspect ratio.
    if img.width > img.height:
        # Image is wider than it is tall.
        new_width = max_size
        new_height = int(max_size * (img.height / img.width))
    else:
        # Image is taller than it is wide.
        new_height = max_size
        new_width = int(max_size * (img.width / img.height))
    
    resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)
    
    # Calculate padding to center the image within a square of size max_size x max_size.
    padding_left   = (max_size - new_width) // 2
    padding_top    = (max_size - new_height) // 2
    padding_right  = max_size - new_width - padding_left
    padding_bottom = max_size - new_height - padding_top
    
    padded_img = ImageOps.expand(resized_img, border=(padding_left, padding_top, padding_right, padding_bottom), fill='black')
    
    print(padded_img.size)

    byte_stream = io.BytesIO()
    padded_img.save(byte_stream, format='PNG')
    byte_stream.seek(0)
    byte_stream.getvalue()
    payload = {
        'image': ('image.png', byte_stream, 'image/png'),
        'seed': (None, '0'),
        'cfg_scale': (None, '2.5'),
        'motion_bucket_id': (None, motion)
    }
    headers = {
        'Authorization': f"Bearer {cs.stability_key}"
    }
    headers2 = {
        'Authorization' : f"Bearer {cs.stability_key}",
        'Accept' : 'application/json'
    }
    response = requests.post(url, files=payload, headers=headers)
    print(response.json()["id"])
    task = asyncio.create_task(getinf(ctx, url, headers2, response.json()["id"], "video.mp4"))

@bot.command()
async def promptupscale(ctx, *, prompt):
    url = "https://api.stability.ai/v2beta/stable-image/upscale/creative"
    await ctx.reply("Please note that this may take a while to generate!")
    if len(ctx.message.attachments) == 0:
        await ctx.send("You must attach an image to this command!")
        return
    padded_img = Image.open(io.BytesIO(await ctx.message.attachments[0].read()))
    if (padded_img.width * padded_img.height) > 1_048_576:
        await ctx.reply("The image you provided is too large! Please provide an image that is less than 1024x1024 or approx 1M pixels.")
        return
    print(type(prompt))
    byte_stream = io.BytesIO()
    padded_img.save(byte_stream, format='PNG')
    byte_stream.seek(0)
    byte_stream.getvalue()
    payload = {
        'image': ("image.png", byte_stream)
    }
    data = {
        'prompt': f"{prompt}",
        'negative_prompt': f"{negitive_prompt}",
        'output_format': 'png',
        'seed': '0'
    }
    headers = {
        'Authorization': f"Bearer {cs.stability_key}"
    }
    headers2 = {
        'Authorization' : f"Bearer {cs.stability_key}",
        'Accept' : 'application/json'
    }
    response = requests.post(url, files=payload, data=data, headers=headers)
    print(response.json())
    print(response.json()["id"])
    if response.status_code == 403:
        await ctx.reply(f"Your image or prompt was flagged by Stability AI's content system, if you belive this was a mistake, try again.\nError Details: {response.text}")
        return
    task = asyncio.create_task(getinf(ctx, url, headers2, response.json()["id"], "image.png"))


@bot.command()
async def inspireme(ctx, timetotal = "1"):
    files = []
    if int(timetotal) > 10:
        await ctx.send("You can only request up to 10 images at a time!")
        return
    if int(timetotal) > 3:
        await ctx.send("This might take a few seconds...")
    for i in range(int(timetotal)):
        url = "http://inspirobot.me/api?generate=true"
        response = requests.get(url)
        print(response.content)
        rand = ''.join(random.choice("abcdefghijklmnopqrstuvwxyzQWERTYUIOPASDFGHJKLZXCVBNM1234567890") for i in range(0, 8))
        if response.status_code == 200:
            try:
                new = str(response.content)[2:-1]
                newreq = requests.get(new)
                file = discord.File(io.BytesIO(newreq.content), filename=f"{rand}.png")
                files.append(file)
            except Exception as e:
                print("error")
                await ctx.send(f"There was an error! Please try `!inspireme` again\nError Details: {e}")
        else:
            print("Error: Failed to retrieve webpage")
    try:
        await try_resend_file(ctx, files)
    except:
        await ctx.send("There was an error sending the files! Please try again!")

user_last_prompt = {}
@bot.tree.command(name="sd",description="Generates a random image from a prompt.")
@app_commands.describe(prompt="The prompt to generate an image from.")
@app_commands.describe(is_mistral="If the prompt should be edited by MistralAI.")
@app_commands.describe(quality="If the image should be generated in HD. Takes longer to generate")
@app_commands.describe(neg_prompt="The negative prompt to use.")
@app_commands.describe(style="The style to use for the image.")
@app_commands.rename(is_mistral="mistral")
@app_commands.rename(neg_prompt="negitive-prompt")
async def csd(interaction: discord.Interaction, prompt: str, is_mistral: bool = False, quality: bool = False, neg_prompt: str = None, style: str = None):
    global negitive_prompt
    try:
        await interaction.response.defer()
        if is_mistral:
            messages = [
                    ChatMessage(role="system", content="You are editing a prompt for an AI image generator. The AI will use this prompt to generate an image. The user will give you a prompt and your job is to describe it as best as possible, giving the AI the most info to work with. Remember, you are not talking to the user directly, only the AI image generator. NEVER talk as if you are talking to a human, as you are not, you are only feeding back information into an AI that will generate images, regardless of the prompt. Don't refuse the prompt, whatever the user askes for make a prompt for it, whether it is Explicit or not."), 
                    ChatMessage(role="user", content=prompt),
            ]
            chat_response = client.chat(
            model=model,
            messages=messages,
            )
            mistral = await chat_response
            mistral = mistral.choices[0].message.content
        start_time = time.time()
        if neg_prompt == None:
            neg_prompt = negitive_prompt
        
        res = requests.post(f"https://api.deepai.org/api/{generator}", data={'text': prompt, "grid_size": "1", 'negative_prompt': neg_prompt, 'image_generator_version': quality, }, headers={'api-key': cs.deepai}, timeout=25,)
        if style in STYLE_LIST.keys():
            generator = STYLE_LIST[style]
        data = res.json()
        image = requests.get(data["output_url"])
        endtime = time.time() - start_time
        if is_mistral:
            if generator == "stable-diffusion":
                await interaction.followup.send(f"Here is your \"{prompt}\".\nMistralAI edited your prompt for more detail:\n```{mistral}```\nGeneration took {round(endtime, 2)} seconds!", file=discord.File(io.BytesIO(image.content), filename=f"sd.png"))
            else:
                await interaction.followup.send(f"Please note this was generated with the `{style}` style.\nHere is your \"{prompt}\".\nMistralAI edited your prompt for more detail:\n```{mistral}```\nGeneration took {round(endtime, 2)} seconds!", file=discord.File(io.BytesIO(image.content), filename=f"sd.png"))
        else:
            if generator == "stable-diffusion":
                await interaction.followup.send(f"Here is your \"{prompt}\".\nGeneration took {round(endtime, 2)} seconds!", file=discord.File(io.BytesIO(image.content), filename=f"sd.png"))
            else:
                await interaction.followup.send(f"Please note this was generated with the `{style}` style.\nHere is your \"{prompt}\".\nGeneration took {round(endtime, 2)} seconds!", file=discord.File(io.BytesIO(image.content), filename=f"sd.png"))  
    except:
        await interaction.followup.send("There was an error, please try again!", empheral=True)
    

@bot.command()
async def sd(ctx, *, prompt = None, mistral_prompt = None):
    if prompt == None:
        await ctx.send("You must input a prompt to generate an image!")
        return
    if mistral_prompt != None:
        prompt, oldprompt = mistral_prompt, prompt
    global curr_style
    global user_last_prompt
    global sd_quality
    global last_sent_message
    global negitive_prompt
    if ctx.author.id not in sd_quality:
        quality = "standard"
    else:
        if sd_quality[ctx.author.id]:
            quality = "hd"
        else:
            quality = "standard"
    if quality == "hd":
        await ctx.reply("Please note that you are using the HD version of Stable Diffusion. This may take a while to generate!")
    user_last_prompt[ctx.author.id] = prompt
    if ctx.author.id in curr_style.keys():
        generator = curr_style[ctx.author.id]
    else:
        generator = curr_style["default"]
    if generator == "stable-diffusion-old":
        async with ctx.typing():
            api = "https://api.stability.ai"
            key = cs.stability_key
            start_time = time.time()
            print(temp)
            try:
                res = requests.post(f"{api}/v1/generation/stable-diffusion-xl-beta-v2-2-2/text-to-image", 
                                headers={
                                "Accept": "application/json",
                                "Authorization": f"Bearer {key}"},
                                json={
                                    "text_prompts": [
                                        {
                                            "text": prompt
                                        }
                                    ],
                                    "cfg_scale": 7,
                                    "samples": 1,
                                    "steps": 30,
                                }, timeout=20,       
            )
                print("done")
            except:
                await ctx.reply("There was an error! Please try again!")
                return
            print(res.status_code)
            if res.status_code != 200:
                if res.status_code == 400:
                    data = res.json()
                    if data["name"] == "invalid_prompts":
                        await ctx.reply("Sorry, but that prompt has words that are not allowed! Please change the prompt and try again!")
                        return
                    else:
                        await ctx.reply(f"There was an error\nError: {res.status_code}\nDetails: {res.text}")
                        return
                else:
                    await ctx.reply(f"Error: {res.status_code}\nDetails: {res.text}")
                    return    
            else:
                data = res.json()
                for i, image in enumerate(data["artifacts"]):
                    endtime = time.time() - start_time
                    try:
                        if image["finishReason"] == "CONTENT_FILTERED":
                            sent = await ctx.reply(f"Sorry, but the image was was blurred because we belive the generation was NSFW.\nHere is your \"{prompt}\".\nGeneration took {round(endtime, 2)} seconds!", file=discord.File(io.BytesIO(base64.b64decode(image["base64"])), filename=f"sd_{i}.png"))
                        else:
                            sent = await ctx.reply(f"Here is your \"{prompt}\".\nGeneration took {round(endtime, 2)} seconds!", file=discord.File(io.BytesIO(base64.b64decode(image["base64"])), filename=f"sd_{i}.png"))
                    except:
                        if image["finishReason"] == "CONTENT_FILTERED":
                            sent = await ctx.reply(f"Sorry, but the image was blurred because we belive the generation was NSFW.\nHere is your \"{prompt}\".\nGeneration took {round(endtime, 2)} seconds!", file=discord.File(io.BytesIO(base64.b64decode(image["base64"])), filename=f"sd_{i}.png"))
                        else:
                            sent = await ctx.reply(f"Here is your \"{prompt}\".\nGeneration took {round(endtime, 2)} seconds!", file=discord.File(io.BytesIO(base64.b64decode(image["base64"])), filename=f"sd_{i}.png"))
                last_sent_message = sent.id
                user_last_prompt[ctx.author.id] = prompt
    else:
        async with ctx.typing():
            
            start_time = time.time()
            try:
                res = requests.post(f"https://api.deepai.org/api/{generator}", data={'text': prompt, "grid_size": "1", 'negative_prompt': negitive_prompt, 'image_generator_version': quality, }, headers={'api-key': cs.deepai}, timeout=25,)
            except requests.exceptions.Timeout:
                await ctx.reply("The server timed out! Please try again later.")
                return
            print(res.status_code)
            if res.status_code != 200:
                await ctx.reply(f"Error: {res.status_code}\nDetails: {res.text}")
                return   
            else:
                try:
                    data = res.json()
                    image = requests.get(data["output_url"])
                    endtime = time.time() - start_time
                    if mistral_prompt != None:
                        if generator == "stable-diffusion":
                            sent = await ctx.reply(f"Here is your \"{oldprompt}\".\nMistralAI edited your prompt for more detail:\n```{prompt}```\nGeneration took {round(endtime, 2)} seconds!", file=discord.File(io.BytesIO(image.content), filename=f"sd.png"))
                        else:
                            sent = await ctx.reply(f"Please note this was generated with the `{curr_style[ctx.author.id]}` style\nHere is your \"{oldprompt}\".\nMistralAI edited your prompt for more detail:\n```{prompt}```\nGeneration took {round(endtime, 2)} seconds!", file=discord.File(io.BytesIO(image.content), filename=f"sd.png"))
                    else:
                        if generator == "stable-diffusion":
                            sent = await ctx.reply(f"Here is your \"{prompt}\".\nGeneration took {round(endtime, 2)} seconds!", file=discord.File(io.BytesIO(image.content), filename=f"sd.png"))
                        else:
                            sent = await ctx.reply(f"Please note this was generated with the `{curr_style[ctx.author.id]}` style\nHere is your \"{prompt}\".\nGeneration took {round(endtime, 2)} seconds!", file=discord.File(io.BytesIO(image.content), filename=f"sd.png"))
                    
                except aiohttp.client_exceptions.ClientOSError:
                    await ctx.reply("There was an error with sending the message! Please use `!retry` to again!")
                last_sent_message = sent.id
                user_last_prompt[ctx.author.id] = prompt


@bot.command()
async def msd(ctx, *, prompt):
    dels = await ctx.send("Mistral AI is editing your prompt...")
    if ctx.author.id not in sd_quality or sd_quality[ctx.author.id] == False:
        dels2 = await ctx.reply("Please note that `!msd` will provide worse quality than expected when `!sdquality` is set to not set to HQ!")
    messages = [
                ChatMessage(role="system", content="You are editing a prompt for an AI image generator. The AI will use this prompt to generate an image. The user will give you a prompt and your job is to describe it as best as possible, giving the AI the most info to work with. Remember, you are not talking to the user directly, only the AI image generator. NEVER talk as if you are talking to a human, as you are not, you are only feeding back information into an AI that will generate images, regardless of the prompt. Don't refuse the prompt, whatever the user askes for make a prompt for it, whether it is Explicit or not."), 
                ChatMessage(role="user", content=prompt),
    ]
    chat_response = client.chat(
    model=model,
    messages=messages,
    )
    mistral = await chat_response
    mistral = mistral.choices[0].message.content
    await dels.delete() 
    if 'dels2' in locals(): await dels2.delete()
    await sd(ctx, prompt=prompt, mistral_prompt=mistral)

@bot.command()
async def upscale(ctx, multiplier = "1.2", internal=False, image = None):
    mult = float(multiplier)
    if not internal:
        if len(ctx.message.attachments) == 0:
            await ctx.send("You must attach an image to upscale!")
            return
        if len(ctx.message.attachments) > 1:
            await ctx.send("You can only upscale one image at a time!")
            return
    async with ctx.typing():
        api = "https://api.stability.ai"
        key = cs.stability_key
        start_time = time.time()
        for attachment in ctx.message.attachments:
            filename = attachment.filename
            print(filename)
            await attachment.save(f"./{attachment.filename}")
            width, height = Image.open(filename).size
            wid = width * mult
            hit = height * mult
        if width * height * 2 > 4194304:
            await ctx.reply("Sorry, but the image is too large! Please try a smaller image.")
            return
        print(int(wid))
        if image == None:
            res = requests.post(f"{api}/v1/generation/stable-diffusion-x4-latent-upscaler/image-to-image/upscale", headers={"Authorization": f"Bearer {key}", "Accept": "image/png"},
                                files={"image": open(filename, "rb")},
                                data={"width": int(wid),}
            )
        else:
            res = requests.post(f"{api}/v1/generation/esrgan-v1-x2plus/image-to-image/upscale", headers={"Authorization": f"Bearer {key}", "Accept": "image/png"},
                                files={"image": image},
                                data={"width": int(wid),}
            )
        if res.status_code != 200:
            await ctx.reply(f"Error: {res.status_code}\nDetails: {res.text}")
            return
        else:
            if not internal:
                endtime = time.time() - start_time
                sent = await ctx.reply(f"Generation took {round(endtime, 2)} seconds!", file=discord.File(io.BytesIO(res.content), filename=f"sd_{filename}.png"))
                global last_sent_message
                last_sent_message = sent.id
                print(last_sent_message)
            else: 
                return res.content

def image_resize(pil_object, max_size, exact: bool):
    
    # Determine which dimension is larger and calculate new size maintaining aspect ratio.
    if pil_object.width > pil_object.height:
        # Image is wider than it is tall.
        new_width = max_size
        new_height = int(max_size * (pil_object.height / pil_object.width))
    else:
        # Image is taller than it is wide.
        new_height = max_size
        new_width = int(max_size * (pil_object.width / pil_object.height))
    
    resized_pil_object = pil_object.resize((new_width, new_height), Image.ANTIALIAS)
    
    # Calculate padding to center the image within a square of size max_size x max_size.
    padding_left   = (max_size - new_width) // 2
    padding_top    = (max_size - new_height) // 2
    padding_right  = max_size - new_width - padding_left
    padding_bottom = max_size - new_height - padding_top
    
    return ImageOps.expand(resized_pil_object, border=(padding_left, padding_top, padding_right, padding_bottom), fill='black')

def pil_to_bytes(image):
    byte_stream = io.BytesIO()
    image.save(byte_stream, format='PNG')
    byte_stream.seek(0)
    return byte_stream.getvalue()
supported_image_formats = [
    'BMP',  # Windows Bitmap
    'DIB',  # Device Independent Bitmap
    'EPS',  # Encapsulated PostScript
    'GIF',  # Graphics Interchange Format
    'ICNS', # Apple Icon Image format
    'ICO',  # Windows Icon
    'IM',   # ImageMagick format
    'JPEG', # Joint Photographic Experts Group, also known as JPG
    'JPG',  # Joint Photographic Experts Group, also known as JPEG
    'JP2',  # JPEG 2000 
    'PCX',  # Personal Computer Exchange
    'PNG',  # Portable Network Graphics
    'PPM',  # Portable Pixel Map (Netpbm)
    'PBM',  # Portable Bit Map (Netpbm)
    'PGM',  # Portable Gray Map (Netpbm)
    'PSD',  # Adobe Photoshop Document format
    'TIFF',
    'TIF'   ,# Tagged Image File Format 
    'WEBP'
]
@bot.command()
async def sortpixels(ctx):
    if len(ctx.message.attachments) == 0:
        await ctx.send("You must attach one or more images to sort!")
        return
    for image in ctx.message.attachments:
        print(image.filename.split(".")[-1].upper())
        if image.filename.split(".")[-1].upper() not in supported_image_formats:
            await ctx.reply("Sorry, but one or more files are not supported! Please try again!")
            return
        img = io.BytesIO(await image.read())
        pilimg = Image.open(img)
        i_bytes = io.BytesIO()
        if pilimg.format == "GIF":
            framesout = []
            frames = [frame.copy() for frame in ImageSequence.Iterator(pilimg)]
            durations = [frame.info['duration'] for frame in frames]
            disposals = [frame.info.get('disposal', 2) for frame in frames]
            for frame in frames:
                # Apply pixel sort to each frame
                processed_frame = pixelsort(frame)
                
                # Append the processed frame to the list of frames
                framesout.append(processed_frame)
            
            # Save frames as a new GIF
            framesout[0].save(
                i_bytes,
                save_all=True,
                append_images=framesout[1:],
                loop=0,  # Loop forever
                disposal=disposals,
                duration=durations,
                format='GIF'
            )
            i_bytes.seek(0)
            await ctx.reply(file=discord.File(i_bytes, filename=f"pixelsorted_{image.filename}.gif"))
        else:
            pilimg = pixelsort(pilimg)
            pilimg.save(i_bytes, format='PNG')
            i_bytes.seek(0)
            await ctx.reply(file=discord.File(i_bytes, filename=f"pixelsorted_{image.filename}.png"))

@bot.command()
async def prompt_variation(ctx, *, prompt = None):
    if prompt == None:
        await ctx.send("You must input a prompt to make a variation of!")
        return
    global temp
    if len(ctx.message.attachments) == 0:
        await ctx.send("You must attach an image to make a variation of!")
        return
    if len(ctx.message.attachments) > 1:
        await ctx.send("You can only make a variation of one image at a time!")
        return
    org_image = await ctx.message.attachments[0].read()
    ret_image = image_resize(Image.open(io.BytesIO(org_image)), (512, 512), True)
    width, height = ret_image.size
    square_image = pil_to_bytes(ret_image)
    if width * height < 262144:
        await upscale(ctx, "2.0", True, square_image)
    async with ctx.typing():
        api = "https://api.stability.ai"
        key = cs.stability_key
        start_time = time.time()
        print(temp)
        try:
            res = requests.post(f"{api}/v1/generation/stable-diffusion-xl-beta-v2-2-2/image-to-image", 
                            headers={
                            "Accept": "application/json",
                            "Authorization": f"Bearer {key}"},
                            files={"init_image": square_image},
                            data={
                            "image_strength": 0.5,
                            "init_image_mode": "IMAGE_STRENGTH",
                            "text_prompts[0][text]": f"{prompt}",
                            "text_prompts[0][weight]": 0.5,
                            "cfg_scale": temp,
                            "clip_guidance_preset": "FAST_GREEN",
                            "samples": 1,
                            "steps": 30,
                            }, timeout=20       
        )
            print("done")
        except:
            await ctx.reply("There was an error! Please try again!")
            return
        print(res.status_code)
        if res.status_code != 200:
            if res.status_code == 400:
                data = res.json()
                if data["name"] == "invalid_prompts":
                    await ctx.reply("Sorry, but that prompt has words that are not allowed! Please change the prompt and try again!")
                    return
                else:
                    await ctx.reply(f"There was an error\nError: {res.status_code}\nDetails: {res.text}")
                    return
            else:
                await ctx.reply(f"Error: {res.status_code}\nDetails: {res.text}")
                return    
        else:
            data = res.json()
            for i, image in enumerate(data["artifacts"]):
                endtime = time.time() - start_time
                try:
                    if image["finishReason"] == "CONTENT_FILTERED":
                        sent = await ctx.reply(f"Sorry, but the image was was blurred because we belive the generation was NSFW.\nHere is your \"{prompt}\".\nGeneration took {round(endtime, 2)} seconds!", file=discord.File(io.BytesIO(base64.b64decode(image["base64"])), filename=f"sd_{i}.png"))
                    else:
                        sent = await ctx.reply(f"Here is your \"{prompt}\".\nGeneration took {round(endtime, 2)} seconds!", file=discord.File(io.BytesIO(base64.b64decode(image["base64"])), filename=f"sd_{i}.png"))
                except:
                    if image["finishReason"] == "CONTENT_FILTERED":
                        sent = await ctx.reply(f"Sorry, but the image was blurred because we belive the generation was NSFW.\nHere is your \"{prompt}\".\nGeneration took {round(endtime, 2)} seconds!", file=discord.File(io.BytesIO(base64.b64decode(image["base64"])), filename=f"sd_{i}.png"))
                    else:
                        sent = await ctx.reply(f"Here is your \"{prompt}\".\nGeneration took {round(endtime, 2)} seconds!", file=discord.File(io.BytesIO(base64.b64decode(image["base64"])), filename=f"sd_{i}.png"))
            global last_sent_message
            last_sent_message = sent.id
            global user_last_prompt
            user_last_prompt[ctx.author.id] = prompt


@bot.command()
async def tpdne(ctx, total = 1):
    if int(total) > 10:
        await ctx.send("You can only request up to 10 images at a time!")
        return
    if int(total) > 3:
        message = await ctx.send("This might take a few seconds...")
    files = []
    for i in range(int(total)):
        api = "https://thispersondoesnotexist.com"
        response = requests.get(api, timeout=5)
        if response.status_code != 200:
            ctx.reply("There was an error! Please try again!")
        else: 
            file=discord.File(io.BytesIO(response.content), filename="tpdne.png")
            files.append(file)
    await ctx.send("This person doesn't exist...",files=files)
    try:
        await message.delete()
    except:
        pass

@bot.command()
async def rn(ctx):
    global resnet
    resnet.eval()

    image_bytes = await ctx.message.attachments[0].read()
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = resnet(input_batch)

    num_predictions = 5
    top5_prob, top5_indices = torch.topk(output, num_predictions)
    top5_prob = torch.nn.functional.softmax(top5_prob, dim=1)[0]
    
    predicted_idx = torch.argmax(output).item()

    class_label = class_labels[predicted_idx]
    probability = torch.nn.functional.softmax(output, dim=1)[0][predicted_idx].item()

    totaloutput = "Classification (Probability)\n"

    for i in range(num_predictions):
        class_label = class_labels[top5_indices[0][i]]
        probability = top5_prob[i].item() * 100
        totaloutput = totaloutput + f"{class_label[:-2][1:]} ({probability:.2f}%)" + "\n"
    
    await ctx.send(totaloutput)

@bot.command()
async def inpaint(ctx, *, prompt = None):
    if prompt == None:
        await ctx.send("You must input a prompt to make a variation of!")
        return
    global temp
    global negitive_prompt
    if len(ctx.message.attachments) == 0:
        await ctx.send("You must attach an image to inpaint!")
        return
    if len(ctx.message.attachments) > 1:
        await ctx.send("You can only impaint one image at a time!")
        return
    org_image = await ctx.message.attachments[0].read()
    img = Image.open(io.BytesIO(org_image))
    ret_image = image_resize(img, 512, True)
    print(ret_image.size)
    print(img.getbands())
    if 'A' not in img.getbands():
        await ctx.send("You must attach an image with an alpha channel!")
        return
    api = "https://api.stability.ai"
    key = cs.stability_key
    start_time = time.time()
    async with ctx.typing():
        try:
            res = requests.post(
            f"{api}/v1/generation/stable-diffusion-xl-beta-v2-2-2/image-to-image/masking",
            headers={
                "Accept": 'application/json',
                "Authorization": f"Bearer {key}"
            },
            files={
                'init_image': io.BytesIO(pil_to_bytes(ret_image)),
            },
            data={
                "mask_source": "INIT_IMAGE_ALPHA",
                "text_prompts[0][text]": prompt,
                "text_prompts[0][weight]": 1,
                "text_prompts[1][text]": negitive_prompt,
                "text_prompts[1][weight]": -1,
                "cfg_scale": temp,
                "clip_guidance_preset": "FAST_BLUE",
                "samples": 1,
                "steps": 30,
            }
            )
        except Exception as e:
                await ctx.reply(f"There was an error! Please try again!\nError Details: {e}")
                return
        print(res.status_code)
        if res.status_code != 200:
            if res.status_code == 400:
                data = res.json()
                if data["name"] == "invalid_prompts":
                    await ctx.reply("Sorry, but that prompt has words that are not allowed! Please change the prompt and try again!")
                    return
                else:
                    await ctx.reply(f"There was an error\nError: {res.status_code}\nDetails: {res.text}")
                    return
            else:
                await ctx.reply(f"Error: {res.status_code}\nDetails: {res.text}")
                return    
        else:
            data = res.json()
            for i, image in enumerate(data["artifacts"]):
                endtime = time.time() - start_time
                try:
                    if image["finishReason"] == "CONTENT_FILTERED":
                        sent = await ctx.reply(f"Sorry, but the image was was blurred because we belive the generation was NSFW.\nHere is your \"{prompt}\".\nGeneration took {round(endtime, 2)} seconds!", file=discord.File(io.BytesIO(base64.b64decode(image["base64"])), filename=f"sd_{i}.png"))
                    else:
                        sent = await ctx.reply(f"Here is your \"{prompt}\".\nGeneration took {round(endtime, 2)} seconds!", file=discord.File(io.BytesIO(base64.b64decode(image["base64"])), filename=f"sd_{i}.png"))
                except:
                    if image["finishReason"] == "CONTENT_FILTERED":
                        sent = await ctx.reply(f"Sorry, but the image was blurred because we belive the generation was NSFW.\nHere is your \"{prompt}\".\nGeneration took {round(endtime, 2)} seconds!", file=discord.File(io.BytesIO(base64.b64decode(image["base64"])), filename=f"sd_{i}.png"))
                    else:
                        sent = await ctx.reply(f"Here is your \"{prompt}\".\nGeneration took {round(endtime, 2)} seconds!", file=discord.File(io.BytesIO(base64.b64decode(image["base64"])), filename=f"sd_{i}.png"))
            global last_sent_message
            last_sent_message = sent.id
            global user_last_prompt
            user_last_prompt[ctx.author.id] = prompt
    
@bot.command()
async def retry(ctx):
    if ctx.author.id not in user_last_prompt:
        await ctx.reply("You must use `!sd` before using `!retry`!")
        return
    else:
        await sd(ctx, prompt=user_last_prompt[ctx.author.id])

@bot.command()
async def sdhelp(ctx):
    await ctx.send("`!sd`\nThis command generates an image based on a prompt.\nUsage: `!sd <prompt>`\nExample: `!sd a cat`\nRequired <prompt>\n----\n`!upscale`\nThis command upscales an image by a multiplier.\nUsage: `!upscale <multiplier>`\nExample: `!upscale 1.2 (multiplier not working)`\nOptional <multiplier>\nRequired <attached image>\n----\n`!inspireme`\nThis command generates an image from InspiroBot.\nUsage: `!inspireme <number of images>`\nExample: `!inspireme 3`\nOptional <number of images>\n----\n`!inpaint`\nUsage: `!inpaint <prompt>`\nExample: `!inpaint Starry Night by Leonardo DaVinci`\nRequired <prompt> <attached image with transparancy>\n----\n`!prompt_variation`\nUsage: `!prompt_variation <prompt>`\nExample: `!prompt_variation a cat`\nRequired <prompt>, <attached image>")

@bot.command()
async def ensurewav(ctx):
    if len(ctx.message.attachments) == 0:
        await ctx.send("You must attach a video/audio to convert!")
        return
    if len(ctx.message.attachments) > 1:
        await ctx.send("You can only convert one video at a time!")
        return
    for attachment in ctx.message.attachments:
        filename = attachment.filename
        extension = filename.split(".")[-1]
        filenamenoext = filename.replace(f".{extension}", "")
        print(filenamenoext)
        print(filename)
        await attachment.save(f"./{filename}")
        try:
            stream = ffmpeg.input(filename)
            stream = ffmpeg.output(stream, f"{filenamenoext}.wav")
            ffmpeg.run(stream)
            if os.path.getsize(f"{filenamenoext}.wav") > 25 * 1024 * 1024:
                remove = await ctx.send("File was too big! Converting to mp3 instead!")
                stream = ffmpeg.input(filename)
                stream = ffmpeg.output(stream, f"{filenamenoext}.mp3")
                ffmpeg.run(stream)
                await try_resend_file(ctx, [discord.File(f"{filenamenoext}.mp3")])
            else:
                await try_resend_file(ctx, [discord.File(f"{filenamenoext}.wav")])
        except Exception as e:
            await ctx.send(f"There was an error! Please try again!\nError Details: {e} {type(e)}")
        finally:
            os.remove(filename)
            os.remove(f"{filenamenoext}.wav")
            if "remove" in locals():
                await remove.delete()
            if os.path.isfile(f"{filenamenoext}.mp3"):
                os.remove(f"{filenamenoext}.mp3")

@bot.command()
async def neg_prompt(ctx, *, neg_prompt = None):
    global negitive_prompt
    if neg_prompt == None:
        await ctx.send(f"The current negitive prompt is\n```{negitive_prompt}```!")
        return
    negitive_prompt = neg_prompt
    await ctx.send(f"Negative prompt set to\n```{negitive_prompt}```!")

@bot.tree.command(description="Rolls a <sided> dice <num> times.")
@app_commands.describe(dice="Total sides of a dice", num="Number of times to roll the dice")
@app_commands.rename(dice="sides", num="rolls")
async def dice(interaction: discord.Interaction, dice: int, num: int):
    if dice == None:
        await interaction.response.send_message("You must input a number! Usage: `!dice <dice sides> <number of rolls(optional)>`")
        return
    if dice == "0":
        await interaction.response.send_message("Your dice must have more than 0 sides!!!! Cannot roll nothing doofus!!!!")
        return
    dice = int(dice) + 1
    roll = np.random.randint(1, dice, num)
    rolls = []
    for i in roll:
        if dice == 3: 
            if i == 1:
                rolls.append("Heads")
            else:
                rolls.append("Tails")
        else:
            rolls.append(i)
    rolls_list = "["
    rolls_list = rolls_list + ', '.join(str(e) for e in rolls)
    rolls_list = rolls_list + "]"
    if len(rolls_list) > 2000:
        await interaction.response.send_message("Sorry, but the reply would go over Discord's character limit so we can't send it! Please reduce the number of dice rolls to help.", empheral=True)
    else:
        await interaction.response.send_message(rolls_list)

@bot.command()
async def choice(ctx, *, choices = None):
    if choices == None:
        await ctx.send("Input your choices in a CSV type list with spaces. Usage: `!choice <choice 1>, <choice 2>, ...`")
        return
    choices = choices.split(", ")
    choice = random.choice(choices)
    await ctx.send(f"Your choice is {choice}!")

@bot.command()
async def delete(ctx, id: int = None):
    global last_sent_message
    if id != None:
        try:
            message = await ctx.fetch_message(id)
            await message.delete()
            await ctx.send("Message deleted.")
        except discord.NotFound:
            await ctx.send("The message was not found, or it was not sent by the bot!")
    elif last_sent_message is not None:
        try:
            message = await ctx.fetch_message(last_sent_message)
            await message.delete()
            await ctx.send("Last message deleted.")
        except discord.NotFound:
            await ctx.send("The last message was not found.")
        last_sent_message = None
    else:
        await ctx.send("No messages sent by the bot yet.")

@bot.command()
async def tile(ctx):
    global tiling
    if ctx.author.id not in tiling:
        tiling[ctx.author.id] = True
        await ctx.send("Tiling enabled!")
        return
    if tiling[ctx.author.id] == False:
        tiling[ctx.author.id] = True
        await ctx.send("Tiling enabled!")
    else:
        tiling[ctx.author.id] = False
        await ctx.send("Tiling disabled!")

@bot.command()
async def sdquality(ctx):
    global sd_quality
    if ctx.author.id not in sd_quality:
        sd_quality[ctx.author.id] = True
        await ctx.send("HQ enabled!")
        return
    if sd_quality[ctx.author.id] == False:
        sd_quality[ctx.author.id] = True
        await ctx.send("HQ enabled!")
    else:
        sd_quality[ctx.author.id] = False
        await ctx.send("HQ disabled!")

@bot.command()
async def sd_temp(ctx, tempurature = None):
    global temp
    if tempurature == None:
        await ctx.send(f"The current tempurature is {temp}!")
        return
    temp_tmp = int(tempurature)
    if temp_tmp > 35:
        await ctx.send("The tempurature must be less than or equal to 35!")
        return
    if temp_tmp < 1:
        await ctx.send("The tempurature must be greater than 1!")
        return
    else: 
        temp = temp_tmp
        await ctx.send(f"Tempurature set to {temp}!")

def make_random_number(number_of_element):
    random_numbers = []
    for i in range(number_of_element):
        random_numbers.append(random.randint(0, 9))
    return random_numbers

@bot.command()
async def print_board(ctx, game_id: int):
    if game_id not in games:
        await ctx.send("Invalid game id!")
        return
    game = games[game_id]
    png = BytesIO()
    png.write(cairosvg.svg2png(bytestring=bytes(game['board']._repr_svg_(), 'utf-8')))
    png.seek(0)
    await ctx.send(file=discord.File(png, filename="chess.png"))

@bot.command()
async def best(ctx, engine, game_id: int, time_limit: int):
    if game_id not in games:
        await ctx.send("Invalid game id!")
        return
    board = games[game_id]['board']

    if engine not in engines:
        await ctx.send("Invalid engine!")
        return

    transport, engine = await chess.engine.popen_uci(engines[engine])
    result = await engine.play(board, chess.engine.Limit(time=time_limit))
    await ctx.send(f"Best move: {result.move}")
    await engine.quit()

@bot.command()
async def eval(ctx, engine, game_id_or_fen):
    if game_id_or_fen.isdigit():
        game_id = int(game_id_or_fen)
        if game_id not in games:
            await ctx.send("Invalid game id!")
            return
        board = games[game_id]['board']
    else:
        board = chess.Board(game_id_or_fen)

    if engine == "all":
        engine_list = list(engines.keys())
    elif engine in engines:
        engine_list = [engine]
    else:
        await ctx.send("Invalid engine!")
        return

    for engine_name in engine_list:
        transport, engine = await chess.engine.popen_uci(engines[engine_name])
        info = await engine.analyse(board, chess.engine.Limit(depth=20, time=5.0))
        score = str(info["score"].relative)
        

        if score.startswith("#"):
            if score.startswith("#+"):
                await ctx.send(f"Checkmate in {(score[2:])} move(s) for White! ({engine_name})")
            else:
                await ctx.send(f"Checkmate in {(score[1:])} move(s) for Black! ({engine_name})")
        else:
            await ctx.send(f"The score is {score[0]}{int(score[1:])/100}! ({engine_name})")

        await engine.quit()

@bot.command()
async def move(ctx, game_id: int, move: str):
    if game_id not in games:
        await ctx.send("Invalid game id!")
        return

    game = games[game_id]
    board = game['board']
    engine = game.get('engine')
    print(engine)

    try:
        board.push_san(move)
    except:
        await ctx.send("Invalid move!")
        return

    # If there is an engine, ask it to play the next move
    if engine is not None:
        result = await engine.play(board, chess.engine.Limit(time=2.0))
        board.push(result.move)

    # Save the game
    game['board'] = board
    games[game_id] = game

    # Send the board
    png = BytesIO()
    png.write(cairosvg.svg2png(bytestring=bytes(board._repr_svg_(), 'utf-8')))
    png.seek(0)
    await ctx.send(file=discord.File(png, filename="chess.png"))

@bot.command()
async def list_engines(ctx):
    engine_list = list(engines.keys())
    await ctx.send(f"Available engines: {', '.join(engine_list)}")

@bot.command()
async def startchess(ctx, player1 = None, player2 = None, forcewhite = None, strength = None, chess960 = None, *options):
    global games
    if player1 is None or player2 is None:
        await ctx.send("You need to specify two players to start a game of chess!\nExample: `!startchess <player1> <player2> <forcewhite(optional)> <strength(optional)> <chess960(optional, not working)>`")
        return
    game_id = random.randint(1000, 9999)
    while game_id in games:
        game_id = random.randint(1000, 9999)
    game = {'game_id': game_id, 'engine': None, 'board': chess.Board()}
    if player1 in engines:
        transport, game['engine'] = await chess.engine.popen_uci(engines[player1])
    elif player2 in engines:
        transport, game['engine'] = await chess.engine.popen_uci(engines[player2])
    if forcewhite is not None:
        if forcewhite == player1:
            game['white'], game['black'] = player1, player2
        elif forcewhite == player2:
            game['white'], game['black'] = player2, player1
    else:
        game['white'], game['black'] = random.sample([player1, player2], 2)
    if game['engine'] and forcewhite == game['engine']:
        if player1 == "stockfish" or player2 == "stockfish":
            if strength is not None or strength != "default":
                game['engine'].configure({"Skill Level": int(strength)})
        result = await game['engine'].play(game['board'], chess.engine.Limit(time=3.0))
        game['board'].push(result.move)
    games[game_id] = game
    png = BytesIO()
    png.write(cairosvg.svg2png(bytestring=bytes(game['board']._repr_svg_(), 'utf-8')))
    png.seek(0)
    await ctx.send(f"Game Made! The game id is {game_id}", file=discord.File(png, filename="chess.png"))
    
@bot.command()
async def deepfry(ctx, color = 2.0, contrast = 2.0, sharpness = 2.0, noise = 1):
    # Open the image file
    files = []
    if len(ctx.message.attachments) == 0:
        await ctx.send("You must attach images to deepfry!")
        return
    for attachment in ctx.message.attachments:
        
        img = Image.open(io.BytesIO(await attachment.read()))
    
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(float(color))
        print(float(color))
    
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(float(contrast))

        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(float(sharpness))

        arr = np.array(np.asarray(img))
        noisy_arr = arr + int(noise) * arr.std() * np.random.random(arr.shape)
    
        noisy_img = Image.fromarray(np.uint8(noisy_arr))
        
        files.append(discord.File(io.BytesIO(pil_to_bytes(noisy_img)), filename=f"deepfry_{attachment.filename}"))
    await try_resend_file(ctx, files)


async def style_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    types = list(STYLE_LIST.keys())
    return [
        app_commands.Choice(name=search_type, value=search_type)
        for search_type in types if current.lower() in search_type.lower()
    ]

@bot.tree.command(description="Changes the sytle of Stable Diffusion, or lists the current style.")
@app_commands.autocomplete(input=style_autocomplete)
async def style(interaction: discord.Interaction, input: str = None):
    global curr_style
    if input == None:
        if interaction.user.id not in curr_style:
            await interaction.send(f"The current style is `{curr_style['default']}`")
            return
        await interaction.send(f"The current style is `{curr_style[interaction.user.id]}`!")
        return
    if input.lower() == "list":
        await interaction.send("Here is a list of supported styles!\n```" + "\n".join(STYLE_LIST.keys()) + "```")
        return
    try:
        curr_style[interaction.user.id] = STYLE_LIST[input.lower()]
        await interaction.send(f"Style set to `{input}` for {interaction.user.display_name}!")
    except KeyError:
        await interaction.send("That style is not supported! Please use `!style list` to see a list of supported styles!")

@bot.command()
async def luhn_check(ctx, card):
    card = card.replace(" ", "")
    if len(card) != 16:
        await ctx.send("The card number must be 16 digits long!")
        return
    card = list(card)
    card = [int(i) for i in card]
    card = [card[i] * 2 if i % 2 == 0 else card[i] for i in range(len(card))]
    card = [i - 9 if i > 9 else i for i in card]
    card = sum(card)
    if card % 10 == 0:
        await ctx.send("The card number is valid!")
    else:
        await ctx.send("The card number is invalid!")

@bot.command()
async def luhn_generate(ctx):
    #chatgpt says the alg works but it does NOT
    random_master_int = make_random_number(13)
    random_master_int.insert(0,5)
    random_master_int.insert(1,4)
    print(random_master_int)
    sum_even = []
    sum_odd = []
    for index,element in enumerate(random_master_int):
        if index % 2 != 0:
            print(index)
            r = random_master_int[index] * 2
            character_string = list(str(r))
            character_int = map(int, character_string)
            sum_even.append(sum(character_int))
        if index % 2 == 0:
            sum_odd.append(element)
    total_sum = sum(sum_even)+sum(sum_odd) * 9
    random_master_int.append(total_sum % 10)
    credit_card_number = ''.join(map(str, random_master_int))
    await ctx.send(f"Here is your card number! {credit_card_number}")

@bot.command()
async def clear(ctx):
    global message_history
    message_history.clear()
    await ctx.send("Conversation cleared!")

@bot.command()
async def mprompt(ctx, *, prompt = None):
    global message_history
    if prompt == None:
        await ctx.send("Current prompt is\n```"+ message_history.prompt.content + "```!")
        return
    message_history.set_prompt(prompt)
    if prompt.lower() == "clear":
        if message_history.prompt != None:
            message_history.prompt = None
            ret = message_history.remove_idx(0)
            print(message_history.prompt)
            print(str(ret))
            await ctx.send("Prompt cleared!")
            await clear(ctx)
        else:
            await ctx.send("There is no prompt to clear!")
    else:
        await ctx.send("Prompt added!")
        await clear(ctx) 

@bot.command()
async def mpromptnc(ctx, *, prompt):
    global message_history
    message_history.set_prompt(prompt)
    await ctx.send("Prompt added!")

@bot.command()
async def print_history(ctx):
    history = message_history.construct_history_print(list=False)
    if len(history) > 2000:
        history = history[:2000]
    await ctx.send(f"{history}")
    if len(history) > 2000:
        await ctx.send(f"Message was cut. Full message is {len(history)} characters long.")

@bot.command()
async def copypasta(ctx):
    global bag
    await ctx.send(bag.draw())

@bot.command()
async def stopstream(ctx):
    global stop
    stop = True

@bot.tree.command(description="Flips a coin an amount of times.")
@app_commands.describe(flips="The amount of times to flip the coin. Default is 1.")
async def coin(interaction: discord.Interaction, flips: int = 1):
    if flips < 1:
        await interaction.response.send_message("You must flip the coin at least once!")
        return
    dice = 3
    roll = np.random.randint(1, dice, flips)
    rolls = []
    for i in roll:
        if dice == 3: 
            if i == 1:
                rolls.append("Heads")
            else:
                rolls.append("Tails")
        else:
            rolls.append(i)
    rolls_list = "["
    rolls_list = rolls_list + ', '.join(str(e) for e in rolls)
    rolls_list = rolls_list + "]"
    if len(rolls_list) > 2000:
        await interaction.response.send_message("Sorry, but the reply would go over Discord's character limit so we can't send it! Please reduce the number of dice rolls to help.", ephemeral=True)
    else:
        await interaction.response.send_message(rolls_list)

@bot.tree.command(description="Sends a message to a specified user.")
@app_commands.describe(user="The user to DM.", message="The message to send.")
async def dm(interaction: discord.Interaction, user: discord.Member, message: str):
    await user.send(message)
    await interaction.response.send_message("DM sent!", ephemeral=True)

async def main():
    global cog
    cog = Music(bot)
    await bot.add_cog(Music(bot))
    await bot.start(cs.bot_token)

async def b_close():
    for cog_name in list(bot.cogs.keys()):
        await bot.remove_cog(cog_name)
    await bot.close()

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("Exception, saving chat memory")
    message_history.save()
    print("memory saved, closing bot")
    asyncio.run(b_close())