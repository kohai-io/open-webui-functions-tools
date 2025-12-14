# Nano Banana Pro - User Guide

**How to Use Google Gemini 3 Pro Image Generation**

This guide helps you get the best results from Nano Banana Pro, a powerful AI image generation tool using Google's Gemini 3 Pro Image model.

## How do I generate an image?

Simply describe what you want to see in natural language. Be as detailed or as simple as you like:

**Basic example:**
```
Create a serene mountain landscape at sunset with pine trees
```

**Detailed example:**
```
Generate a photorealistic portrait of an elderly fisherman with weathered hands, 
standing on a wooden dock at golden hour, dramatic side lighting, cinematic composition
```

## What resolutions are available?

Nano Banana Pro supports three resolution levels. Just include the resolution keyword in your prompt:

- **1K** - Fast generation, good for quick iterations and testing ideas
- **2K** - Balanced quality and speed, recommended for most uses
- **4K** - Maximum detail and quality, best for final outputs (takes longer to generate)

**Example:**
```
Generate a 4K portrait of a cyberpunk cityscape at night
Create a 2K illustration of a fantasy dragon
```

## What aspect ratios can I use?

Nano Banana Pro supports 10 different aspect ratios. You can specify them using natural language or exact ratios:

**Square formats:**
- `1:1` - Use keywords: "square", "1:1", "square format"

**Portrait/Vertical formats:**
- `2:3` - Classic photo portrait
- `3:4` - Standard portrait
- `4:5` - Instagram portrait
- `9:16` - Phone screen, vertical video
- Use keywords: "portrait", "vertical", "phone wallpaper", "tall"

**Landscape/Horizontal formats:**
- `3:2` - Classic photography
- `4:3` - Standard landscape
- `5:4` - Wide landscape
- `16:9` - Widescreen, YouTube
- `21:9` - Ultrawide, cinematic
- Use keywords: "landscape", "horizontal", "widescreen", "ultrawide", "panoramic", "cinematic"

**Examples:**
```
Create a portrait format 2K image of a fashion model
Generate a widescreen 16:9 panoramic ocean view in 4K
Make a square 1:1 profile picture of a golden retriever
Create an ultrawide cinematic shot of a desert highway
```

## Do aspect ratio and resolution settings persist?

Yes! Once you specify a resolution or aspect ratio, it becomes "sticky" and applies to all future images in the conversation until you change it. This saves you from repeating settings every time.

**Example conversation:**
```
You: Create a 4K landscape 16:9 image of mountains
[Image generated in 4K, 16:9 format]

You: Now create a city skyline
[Image automatically generated in 4K, 16:9 format - settings remembered!]

You: Make it portrait format instead
[New setting applied - now 4K, portrait format]
```

## Can I refine or iterate on generated images?

Yes! Nano Banana Pro remembers up to **14 previous images** in your conversation, so you can build on them iteratively. Just describe what changes you want.

**Simple refinements:**
```
You: Create a Victorian mansion
[Image generated]

You: Add a full moon in the sky
[Updated image with moon]

You: Change it to autumn with falling leaves
[Updated with autumn theme]

You: Make it more gothic and mysterious
[Final refined version]
```

**Style transformations:**
```
You: Same image but in watercolor style
You: Generate this in anime art style
You: Transform to pencil sketch
You: Make it photorealistic
You: Convert to vintage 1950s photograph style
```

**Compositional changes:**
```
You: Show the same subject from a bird's eye view
You: Create a close-up version focusing on the face
You: Zoom out to show the full scene with environment
You: Change the camera angle to low perspective
You: Add dramatic backlighting
```

## How many reference images can I use at once?

Nano Banana Pro can reference up to **14 images** from your conversation history. This is unique to Gemini 3 Pro Image and enables:

- **Multi-reference generation**: "Combine the composition from image 1 with the color palette from image 3"
- **Consistent character/object generation**: Reference multiple angles of the same subject
- **Style mixing**: Blend elements from different reference images
- **Complex scene building**: Build up a scene piece by piece across multiple generations

## Can I edit or modify existing images?

Yes! Upload any image and ask for specific modifications:

**Color and tone adjustments:**
```
Make the colors more vibrant and saturated
Convert to black and white with high contrast
Add a warm vintage film look
Increase brightness and make it more cheerful
```

**Background changes:**
```
Change the background to a beach scene
Replace the background with a solid color
Blur the background for shallow depth of field
```

**Content modifications:**
```
Remove the text overlay
Add rain and wet reflections to this street scene
Change it from day to night
Add motion blur to suggest movement
```

**Style transfers:**
```
Transform this photo into an oil painting
Make it look like a comic book illustration
Convert to a minimal flat design
```

## How do I write effective prompts?

**The key is specificity.** The more visual details you provide, the better the results.

❌ **Vague prompt:**
```
Create a city
```

✅ **Specific prompt:**
```
Create a futuristic neon-lit Tokyo street at night with rain reflections, 
cyberpunk aesthetic, crowds with umbrellas, purple and blue color scheme, 4K
```

## What details should I include in my prompts?

Include as many of these elements as relevant:

**Subject** - What's the main focus?
- "A female astronaut", "A vintage sports car", "A fantasy dragon"

**Style** - What artistic approach?
- Photorealistic, oil painting, watercolor, digital illustration, 3D render, anime, comic book, pencil sketch, minimalist, baroque

**Mood/Atmosphere** - What feeling?
- Dark and moody, cheerful and bright, mysterious, dramatic, serene, energetic, melancholic, whimsical

**Lighting** - How is it lit?
- Golden hour, studio lighting, moonlight, neon lights, dramatic shadows, soft diffused light, backlighting, rim lighting, volumetric fog

**Colors** - What palette?
- Vibrant colors, muted tones, monochrome, warm tones, cool tones, "purple and gold color scheme", pastel colors, high contrast

**Composition** - Camera angle and framing?
- Close-up portrait, wide shot, aerial view, low angle, bird's eye view, centered composition, rule of thirds, symmetrical

**Environment/Setting** - Where is it?
- Forest clearing, urban alley, underwater cave, desert dunes, space station, Victorian study, modern kitchen

**Details** - Specific elements?
- Textures (rough, smooth, weathered), weather (rain, fog, snow), time of day, clothing details, architectural elements

## Can you show me an example of a well-structured prompt?

Here's a detailed prompt that includes multiple elements:

```
Generate a 2K portrait format image of a steampunk airship captain in Victorian attire, 
dramatic side lighting from the left, bronze and copper color scheme, 
leather jacket with brass buttons and goggles resting on forehead, 
weathered face with grey beard, cloudy sky background with airship silhouette, 
cinematic composition with shallow depth of field, photorealistic style with slight film grain
```

This prompt specifies:
- ✅ Resolution (2K) and aspect ratio (portrait)
- ✅ Subject (airship captain)
- ✅ Style (steampunk, photorealistic)
- ✅ Lighting (dramatic side lighting)
- ✅ Colors (bronze and copper)
- ✅ Details (clothing, facial features, background)
- ✅ Composition (cinematic, shallow depth of field)
- ✅ Atmosphere (film grain for character)

## What is C2PA and how does it work?

C2PA (Coalition for Content Provenance and Authenticity) is a cryptographic signing system for images. When enabled by your administrator, all generated images include tamper-proof metadata that proves:

- **Source**: Generated by AI (Gemini 3 Pro Image via Nano Banana Pro)
- **Creator**: Your identity/username
- **Timestamp**: Exact date and time of generation
- **AI Training Policy**: Usage restrictions for AI training datasets

This metadata travels with the image and can be verified by C2PA-compatible tools, helping combat misinformation and ensuring proper attribution.

## Does the system suggest follow-up prompts?

Yes! After generating an image, Nano Banana Pro can automatically suggest creative variations and next steps based on what you created. These suggestions are tailored to image generation workflows and might include:

- Style variations ("same subject in watercolor style")
- Compositional changes ("close-up version", "aerial view")
- Lighting/mood variations ("at sunset", "in fog")
- Setting changes ("in winter", "underwater version")
- Related subjects or scenes

## What's the best workflow for creating images?

**1. Start simple, then refine**
- Generate a base image with basic description
- Then iterate with specific changes and refinements
- This is faster than trying to get everything perfect in one prompt

**2. Use appropriate resolution for the task**
- **1K**: Quick iterations, testing concepts, rough drafts
- **2K**: Most general purposes, good balance
- **4K**: Final outputs, print quality, detailed work

**3. Build incrementally**
- Get the composition right first
- Then adjust colors and lighting
- Finally refine details and atmosphere

**4. Leverage reference images**
- Upload examples of styles you like
- Reference multiple images in your conversation history
- Use phrases like "similar to the previous image but..."

**5. Experiment with combinations**
- Mix different art styles ("photorealistic with anime influence")
- Combine time periods ("Victorian architecture with cyberpunk neon")
- Blend aesthetics ("minimalist composition with baroque details")

**6. Be patient with complex prompts**
- Detailed 4K images take longer to generate
- Multiple reference images increase processing time
- Complex compositions may need more generation time

## Why am I getting "No prompt provided" error?

This happens when:
- You send a message without describing what image you want
- You upload an image without instructions on how to modify it

**Solution**: Always include a text description or editing instruction.

❌ Wrong: [Upload image with no text]
✅ Correct: [Upload image] "Make this brighter and add a sunset sky"

## Why is generation taking so long?

Several factors affect generation time:

**Resolution**:
- 1K: Fast (usually under 30 seconds)
- 2K: Moderate (30-60 seconds)
- 4K: Slow (can take 2+ minutes)

**Reference images**:
- More reference images = longer processing
- 14 images will take longer than 1 image

**Complexity**:
- Simple subjects generate faster
- Complex scenes with many elements take longer

**Solution**: Use 2K for faster iteration, save 4K for final outputs.

## What if I get an authentication error?

Authentication errors mean the system can't connect to Google's AI services. This requires:
- Valid Google Cloud Project ID (for Vertex AI mode)
- OR valid Google AI API key (for API mode)

Contact your Open WebUI administrator to configure credentials.

## What image formats are supported?

**Generated images (output)**:
- JPEG and PNG formats
- Automatically uploaded to Open WebUI storage
- Displayed inline in the chat

**Reference images (input)**:
- Any format Open WebUI supports
- Upload via the attachment button
- Can be used for editing or style reference

## Can I generate images without text, just from a reference image?

No, you need to provide at least some instruction. However, it can be very simple:

```
[Upload image] Enhance this
[Upload image] Improve the quality
[Upload image] Make it more dramatic
```

The system needs direction on what changes or enhancements you want.

## What model is Nano Banana Pro using?

By default, Nano Banana Pro uses **gemini-3-pro-image-preview**, Google's professional-grade image generation model with:
- High-resolution output (up to 4K)
- 10 aspect ratio options
- Up to 14 reference images
- Advanced visual reasoning

An alternative faster model (**gemini-2.5-flash-image**) may be available depending on your administrator's configuration.

---

**Document Version**: 3.4  
**Last Updated**: December 2025  
**Pipeline**: nano_banana_pro.py
