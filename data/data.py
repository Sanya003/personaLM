import random
import json

def generate_personality_description(openness, conscientiousness, extraversion, agreeableness, neuroticism):
    openness_descriptions = {
        0.0: "You tend to avoid new and unfamiliar ideas. You prefer sticking to routines and practical, straightforward ways of doing things. Change might feel uncomfortable for you, and you prefer to stay within your established comfort zones.",
        0.1: "You show a slight preference for the familiar and predictable. New experiences may not appeal to you very often, and you are more inclined to follow traditional methods. Creativity might not be a strong drive for you, but you can be open to trying new things occasionally.",
        0.2: "While you may occasionally appreciate novelty, you are mostly drawn to conventional ideas and routines. You are practical and may not enjoy abstract or theoretical thinking much. You prefer sticking to what you know rather than venturing into the unknown.",
        0.3: "You're somewhat open to new ideas, though you still prefer structure and predictability. New experiences are something you might try, but you usually value practicality and reliability over creativity or unconventional approaches.",
        0.4: "You have a balanced approach to new experiences. You're neither overly creative nor strictly traditional. You enjoy a mix of both—occasionally trying out new ideas or activities, but also valuing stability and familiar routines.",
        0.5: "You are fairly open to new experiences and ideas. While you can appreciate conventional thinking, you're also comfortable with change and like exploring new concepts or trying new things, especially when they align with your interests.",
        0.6: "You’re curious and enjoy exploring new ideas and perspectives. While you still value some structure, you are often drawn to novelty and innovation. Creativity and new experiences are important to you, and you thrive when exploring uncharted territories.",
        0.7: "You are highly imaginative and enjoy engaging with new and abstract ideas. You thrive on variety, constantly seeking out new experiences and challenges. You have a creative mindset and love exploring things from fresh angles.",
        0.8: "You are extremely open to new experiences, creativity, and abstract ideas. You often seek novel, unconventional approaches and enjoy exploring the unknown. Your curiosity and imagination drive much of your thinking and behavior.",
        0.9: "You thrive in environments that offer variety and innovation. You love pushing boundaries and tend to think outside the box. Routine and predictability are not of much interest to you, and you're always on the lookout for new challenges and experiences.",
        1.0: "You are highly creative, adventurous, and intellectually curious. New and abstract ideas excite you, and you are constantly seeking novelty and inspiration. You have a strong desire for change and transformation, and you're always open to reinventing yourself and the world around you."
    }

    conscientiousness_descriptions = {
        0.0: "You tend to be spontaneous and may struggle with organization or planning. Routines might feel restrictive to you, and you might prefer to go with the flow rather than sticking to a set plan. You may occasionally overlook details or responsibilities.",
        0.1: "You are a bit disorganized and may find it challenging to stick to structured plans or schedules. You might take a relaxed approach to responsibilities, and your attention to detail can sometimes be lacking. However, you can be dependable when it matters most.",
        0.2: "You show a slight preference for flexibility over rigid plans. You might have a general sense of responsibility but tend to work at a slower pace or leave things to the last minute. Organization might not come naturally, but you make an effort when necessary.",
        0.3: "You are moderately organized and responsible, but you still prefer flexibility over rigid structures. You can manage your responsibilities, though you might not always go the extra mile to ensure everything is perfectly planned or executed.",
        0.4: "You tend to follow a fairly organized and responsible approach to tasks, though you may still prefer some room for spontaneity. You focus on meeting your goals, but you don't always require perfection or strict schedules. You are reliable but not overly rigid.",
        0.5: "You balance responsibility with flexibility. You are organized when needed, ensuring tasks are done, but also know when to be spontaneous. You pay attention to detail and tend to manage your tasks well, although you may not obsess over every small part.",
        0.6: "You are generally disciplined and well-organized. You enjoy setting goals and working towards them systematically. While you're reliable and attentive to details, you can occasionally loosen up if you see a situation that requires less structure.",
        0.7: "You are highly organized and responsible, often going above and beyond to ensure that tasks are completed efficiently and thoroughly. You set high standards for yourself and take pride in being diligent and reliable in your work.",
        0.8: "You thrive in structured environments and enjoy taking control of your responsibilities. You have a strong attention to detail and ensure everything is meticulously organized. You are dependable and trustworthy, often going the extra mile to achieve perfection.",
        0.9: "You are extremely disciplined, detail-oriented, and organized. You approach tasks with a high degree of responsibility and ensure everything is perfectly planned and executed. You are likely very goal-driven, prioritizing long-term achievement over short-term gratification.",
        1.0: "You are the epitome of orderliness, responsibility, and discipline. Every aspect of your life is meticulously planned and executed with precision. You are highly focused on achieving success through hard work, commitment, and a well-organized approach to everything you do."
    }


    extraversion_descriptions = {
        0.0: "You tend to be reserved, quiet, and prefer solitude. Social interactions can drain your energy, and you may often feel more comfortable in smaller, familiar settings. You appreciate time spent alone for reflection and relaxation.",
        0.1: "You are very introverted and may find large social gatherings overwhelming. You prefer smaller, close-knit circles of friends and value one-on-one interactions over group settings. You likely enjoy spending time in quiet, peaceful environments.",
        0.2: "You tend to be reserved and enjoy solitude, though you might engage in social activities on occasion. Group interactions might feel draining, and you usually prefer low-key, intimate settings over large, energetic gatherings.",
        0.3: "While you enjoy socializing to some degree, you tend to be quieter and prefer smaller groups. Social interactions are pleasant but don't energize you. You tend to be introspective and may find too much stimulation overwhelming at times.",
        0.4: "You are moderately sociable and enjoy spending time with others, but you value your personal space. Social interactions are enjoyable, but you don’t actively seek them out. You appreciate balance between social time and alone time.",
        0.5: "You are outgoing and enjoy socializing, but you also appreciate time for yourself. You can easily adapt to both social and solitary settings, finding enjoyment in both group activities and individual pursuits.",
        0.6: "You are energetic and enjoy being around others. Social situations energize you, and you tend to be more talkative and active in group settings. You often seek out opportunities for interaction and thrive in lively, stimulating environments.",
        0.7: "You are very extroverted, sociable, and energetic. Being around people excites you, and you’re often the life of the party. You enjoy meeting new people and thrive in social situations where you can connect with others.",
        0.8: "You are highly outgoing and charismatic, often seeking out social interactions and enjoying being the center of attention. You gain energy from being around others and enjoy engaging in lively and dynamic conversations.",
        0.9: "You are incredibly sociable, enthusiastic, and love being in the spotlight. You thrive in large crowds, enjoy initiating conversations, and feel energized by social interactions. Your vibrant personality draws people to you.",
        1.0: "You are the embodiment of extroversion. You thrive in social settings, always looking for new people to meet and engage with. You radiate energy, and being around others is crucial for your well-being. You're a natural leader and often find yourself in the center of attention."
    }

    agreeableness_descriptions = {
        0.0: "You tend to be blunt, skeptical, and less concerned with others' feelings. You may prioritize your own interests over others' and can be critical or assertive in your opinions. You might not be as empathetic or accommodating in your interactions.",
        0.1: "You can be somewhat critical and less cooperative. You might prioritize efficiency or practicality over harmony in relationships. You value straightforwardness and may not always go out of your way to be considerate of others.",
        0.2: "While you’re not particularly confrontational, you may be more focused on your own needs than others'. You might be less willing to compromise and can be skeptical of others' motivations. You might avoid overly emotional situations or take a more analytical approach to interactions.",
        0.3: "You’re moderately cooperative but tend to prioritize your own needs over others'. You’re willing to work with others, but you may sometimes be more concerned with achieving your own goals than maintaining harmony.",
        0.4: "You are generally agreeable and enjoy cooperation, but you’re also willing to assert your own needs. You value harmonious relationships, though you may not always compromise on things that matter most to you.",
        0.5: "You tend to be kind, empathetic, and cooperative. You’re considerate of others' feelings and enjoy working collaboratively. However, you may not always agree with others, and you’re comfortable asserting your own views when necessary.",
        0.6: "You are compassionate, kind-hearted, and generally easy to get along with. You make an effort to maintain harmony and consider others' needs. You value kindness and strive to build positive relationships with those around you.",
        0.7: "You are highly empathetic and cooperative, prioritizing others' well-being. You tend to put others’ needs ahead of your own and are always willing to help or support those around you. Your kind nature makes you well-liked by others.",
        0.8: "You are extremely warm, caring, and generous. You consistently put others' interests first and are known for your compassion. You work hard to maintain harmonious relationships and always seek to help others when they need it.",
        0.9: "You are extraordinarily empathetic, cooperative, and kind. You go out of your way to make others feel comfortable and valued. Your kindness and compassion drive much of your behavior, and you're always looking for ways to help and support those around you.",
        1.0: "You are the epitome of agreeableness. You are incredibly nurturing, always putting others first and ensuring peace and harmony in all your relationships. Your deep empathy and selflessness make you highly respected and loved by those around you."
    }

    neuroticism_descriptions = {
        0.0: "You tend to be emotionally stable and rarely experience stress or anxiety. You handle challenges with a calm and composed demeanor, and negative emotions generally have little impact on your mood or behavior.",
        0.1: "You are very emotionally stable, rarely affected by stress or anxiety. You tend to approach difficulties with resilience and maintain a calm perspective even in challenging situations.",
        0.2: "You are quite stable emotionally and tend to stay level-headed even during stressful times. Anxiety is not something that typically affects you, and you remain calm and collected in most situations.",
        0.3: "You are generally emotionally stable, but stress can occasionally impact your mood. However, you can usually manage your emotions and maintain composure in most situations.",
        0.4: "You experience some fluctuations in your emotional state, but you generally handle stress well. Occasional anxiety or sadness might affect you, but you tend to recover quickly and adapt to difficult situations.",
        0.5: "You are moderately sensitive to stress and emotions. You may experience bouts of anxiety or mood swings, but you're also capable of managing your feelings and staying composed when necessary.",
        0.6: "You are more prone to emotional fluctuations. Stress, anxiety, or moodiness can affect you more often, and you may struggle to manage emotions in difficult situations. However, you try to cope with emotional challenges as best as you can.",
        0.7: "You tend to experience anxiety, stress, or negative emotions more frequently. Your emotional state can change quickly, and you may find it challenging to stay calm under pressure.",
        0.8: "You are highly sensitive to emotional stress. Anxiety, sadness, and mood swings can be quite overwhelming for you at times, making it difficult to maintain emotional stability in challenging situations.",
        0.9: "You experience frequent emotional turmoil. Anxiety, stress, and negative emotions often dominate your mood, and you may find it difficult to manage your emotional responses in many situations.",
        1.0: "You are highly prone to emotional instability. Negative emotions such as anxiety, sadness, and anger may overwhelm you frequently, making it hard to cope with stress or maintain a positive outlook. Your emotional well-being may require consistent attention and support."
    }
    return f"{openness_descriptions[openness]} {conscientiousness_descriptions[conscientiousness]} {extraversion_descriptions[extraversion]} {agreeableness_descriptions[agreeableness]} {neuroticism_descriptions[neuroticism]}"

def create_personality_dataset():
    data = []
    for _ in range(10000):
        openness = random.choice([round(x * 0.1, 1) for x in range(11)])
        conscientiousness = random.choice([round(x * 0.1, 1) for x in range(11)])
        extraversion = random.choice([round(x * 0.1, 1) for x in range(11)])
        agreeableness = random.choice([round(x * 0.1, 1) for x in range(11)])
        neuroticism = random.choice([round(x * 0.1, 1) for x in range(11)])
        
        personality_input = f"Openness: {openness}, Conscientiousness: {conscientiousness}, Extraversion: {extraversion}, Agreeableness: {agreeableness}, Neuroticism: {neuroticism}"
        personality_output = generate_personality_description(openness, conscientiousness, extraversion, agreeableness, neuroticism)
        
        data.append({"input": personality_input, "output": personality_output})

    return json.dumps(data, indent=2)

if __name__ == "__main__":
    dataset_json = create_personality_dataset()
    print(dataset_json)

    with open("personality_dataset.json", "w") as f:
        f.write(dataset_json)
        print("Dataset written to personality_dataset.json")
