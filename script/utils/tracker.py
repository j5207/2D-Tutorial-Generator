import itchat

@itchat.msg_register(itchat.content.TEXT)
def text_reply(msg):
    return "this is the auto reply from itchat python module"

# print(itchat.search_friends())

itchat.auto_login(hotReload=True)
itchat.run()